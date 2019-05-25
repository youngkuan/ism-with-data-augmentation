# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import os
import shutil

import torch
import torch.nn as nn

from datasets import get_gan_loaders, get_loader
from evaluation import validate
from modules.discrminator import STAGE1_D, STAGE2_D, MatchDiscriminator
from modules.generator import STAGE1_G, STAGE2_G
from utils import deserialize_vocab
from utils import weights_init, save_img_results

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


class GANTrainer(object):
    """
    通过训练GAN，训练图像生成器和判别器
    """

    def __init__(self, arguments):
        self.epochs = arguments['epochs']
        self.batch_size = arguments['batch_size']
        self.num_workers = arguments['num_workers']
        self.learning_rate = arguments['learning_rate']
        self.beta1 = arguments['beta1']
        self.lr_decay_step = arguments['lr_decay_step']
        self.gpus = arguments["gpus"]

        self.model_save_path = arguments['model_save_path']
        self.synthetic_region_path = arguments['synthetic_region_path']
        self.synthetic_image_path = arguments['synthetic_image_path']
        self.r = arguments['r']
        self.arguments = arguments

        self.data_dir = arguments['data_dir']
        self.voc_file = arguments['voc_file']
        self.train_path = os.path.join(self.data_dir, "train")
        # Load Vocabulary Wrapper
        print "----------------load vocabulary--------------"
        vocab = deserialize_vocab(self.train_path, self.voc_file)
        arguments['vocab_size'] = len(vocab)

        self.generator_data_loader = get_gan_loaders(arguments, vocab, self.batch_size,
                                                     self.num_workers)

        # 图像生成器
        self.region_generator = STAGE1_G(arguments).cuda()
        self.region_generator.apply(weights_init)

        self.image_generator = STAGE2_G(arguments).cuda()
        self.image_generator.apply(weights_init)

        # 合成图像判别器
        self.region_discriminator = STAGE1_D(arguments).cuda()
        self.image_discriminator = STAGE2_D(arguments).cuda()

        # 优化器
        self.region_generator_optimizer = \
            torch.optim.Adam(self.region_generator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))
        self.image_generator_optimizer = \
            torch.optim.Adam(self.image_generator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.region_discriminator_optimizer = \
            torch.optim.Adam(self.region_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))
        self.image_discriminator_optimizer = \
            torch.optim.Adam(self.image_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

    def update_regionG(self, regions, captions, lengths, criterion, epoch, iteration):
        """
        更新图像区域生成过程
        :param regions:
        :param captions:
        :param lengths:
        :param criterion:
        :param epoch:
        :param iteration:
        :return:
        """

        # 更新判别器
        self.region_discriminator_optimizer.zero_grad()

        fake_regions, penal, hidden, mu, BM = self.region_generator(captions, lengths)

        region_feature, fake_region_scores = nn.parallel.data_parallel(self.region_discriminator, fake_regions,
                                                                       self.gpus)

        region_feature, real_region_scores = nn.parallel.data_parallel(self.region_discriminator, regions, self.gpus)

        fake_region_labels = torch.FloatTensor(fake_region_scores.size()).fill_(0).cuda()
        real_region_labels = torch.FloatTensor(real_region_scores.size()).fill_(1).cuda()

        errD_fake = criterion(fake_region_scores, fake_region_labels)
        errD_real = criterion(real_region_scores, real_region_labels)
        errD = errD_real.sum() + errD_fake.sum() + penal

        errD.backward(retain_graph=True)
        self.region_discriminator_optimizer.step()

        # 更新生成器
        self.region_generator_optimizer.zero_grad()

        fake_regions, penal, hidden, mu, BM = self.region_generator(captions, lengths)

        region_feature, fake_region_scores = nn.parallel.data_parallel(self.region_discriminator, fake_regions,
                                                                       self.gpus)
        errG_fake = criterion(fake_region_scores, real_region_labels)
        errG = errG_fake.sum() + penal

        errG.backward(retain_graph=True)
        self.region_generator_optimizer.step()

        # save model
        if iteration % 1000 == 0:
            region_g_file = os.path.join(self.model_save_path, '%d_region_g.pth.tar' % epoch)
            save_checkpoint(self.region_generator.state_dict(), True, region_g_file, prefix='')
            region_d_file = os.path.join(self.model_save_path, '%d_region_d.pth.tar' % epoch)
            save_checkpoint(self.region_discriminator.state_dict(), True, region_d_file, prefix='')

        regions = regions.view(-1, regions.size(2), regions.size(3), regions.size(4))
        fake_regions = fake_regions.view(-1, fake_regions.size(2), fake_regions.size(3), fake_regions.size(4))
        save_img_results(regions, fake_regions, epoch, self.synthetic_region_path, self.batch_size)
        print("Epoch: %d, iteration: %d, STAGE I , errD: %f, errG: %f" % (epoch, iteration, errD.data, errG.data))

    def update_imageG(self, images, captions, lengths, criterion, epoch, iteration):
        """
        更新图像生成过程
        :param images:
        :param captions:
        :param lengths:
        :param criterion:
        :param epoch:
        :param iteration:
        :return:
        """
        # 更新判别器
        self.image_discriminator_optimizer.zero_grad()

        fake_images, penal, hidden, mu = self.image_generator(captions, lengths)

        fake_image_scores = nn.parallel.data_parallel(self.image_discriminator, fake_images, self.gpus)
        real_image_scores = nn.parallel.data_parallel(self.image_discriminator, images, self.gpus)

        fake_image_labels = torch.FloatTensor(fake_image_scores.size()).fill_(0).cuda()
        real_image_labels = torch.FloatTensor(real_image_scores.size()).fill_(1).cuda()

        errD_fake = criterion(fake_image_scores, fake_image_labels)
        errD_real = criterion(real_image_scores, real_image_labels)
        errD = errD_real.sum() + errD_fake.sum() + penal

        errD.backward(retain_graph=True)
        self.image_discriminator_optimizer.step()

        # 更新生成器
        self.image_generator_optimizer.zero_grad()

        fake_images, penal, hidden, mu = self.image_generator(captions, lengths)

        fake_image_scores = nn.parallel.data_parallel(self.image_discriminator, fake_images, self.gpus)
        errG_fake = criterion(fake_image_scores, real_image_labels)
        errG = errG_fake.sum() + penal

        errG.backward(retain_graph=True)
        self.image_generator_optimizer.step()

        save_img_results(images, fake_images, epoch, self.synthetic_image_path, self.batch_size)
        print("Epoch: %d, iteration: %d, STAGE II, errD: %f, errG: %f" % (epoch, iteration, errD.data, errG.data))

    def train(self):
        """
        训练图像区域生成器
        :return:
        """
        bce_loss = nn.BCELoss(reduce=False)
        for epoch in range(self.epochs):
            if epoch % self.lr_decay_step == 0 and epoch > 0:
                self.learning_rate *= 0.5
                for param_group in self.image_generator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in self.stack_fake_image_discriminator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            # hidden = self.image_generator.txt_enc.init_hidden(self.batch_size)

            for i, train_data in enumerate(self.generator_data_loader):

                images, regions, captions, lengths, indexes = train_data
                if torch.cuda.is_available():
                    images = images.cuda()
                    regions = regions.cuda()
                    captions = captions.cuda()

                self.update_regionG(regions, captions, lengths, bce_loss, epoch, i)
                self.update_imageG(images, captions, lengths, bce_loss, epoch, i)


class MTrainer(object):
    """
    训练图文匹配模型
    """

    def __init__(self, arguments):
        self.epochs = arguments['epochs']
        self.batch_size = arguments['batch_size']
        self.learning_rate = arguments['learning_rate']

        self.model_save_path = arguments['model_save_path']
        self.region_g_path = os.path.join(self.model_save_path, "0_region_g.pth.tar")
        self.region_d_path = os.path.join(self.model_save_path, "0_region_d.pth.tar")
        self.arguments = arguments

        self.data_dir = arguments['data_dir']
        self.voc_file = arguments['voc_file']
        self.train_path = os.path.join(self.data_dir, "train")
        # Load Vocabulary Wrapper
        print "----------------load vocabulary--------------"
        vocab = deserialize_vocab(self.train_path, self.voc_file)
        arguments['vocab_size'] = len(vocab)

        # self.train_loader = get_loader(arguments, vocab, 'train')

        self.val_loader = get_loader(arguments, vocab, 'dev')
        # 加载预训练生成器
        print "load region generator"
        self.region_generator = STAGE1_G(arguments).cuda()
        self.region_generator.load_state_dict(torch.load(os.path.join(self.region_g_path)))

        print "load region discriminator"
        self.region_discriminator = STAGE1_D(arguments).cuda()
        self.region_discriminator.load_state_dict(torch.load(os.path.join(self.region_d_path)))

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments, self.region_generator,
                                                      self.region_discriminator).cuda()

    def train(self):
        """
        训练
        :return:
        """
        best_rsum = 0
        for epoch in range(self.epochs):
            for i, sample in enumerate(self.val_loader):
                images, captions, lengths, indexes = sample

                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()

                self.match_discriminator.train_emb(images, captions, lengths, epoch)

        # evaluate on validation set
        rsum = validate(self.arguments, self.val_loader, self.match_discriminator)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(self.arguments['model_save_path']):
            os.mkdir(self.arguments['model_save_path'])

        save_checkpoint({
            'epoch': epoch + 1,
            'model': self.match_discriminator.state_dict(),
            'best_rsum': best_rsum,
            'opt': self.arguments,
            'Eiters': self.match_discriminator.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=self.arguments['model_save_path'] + '/')
