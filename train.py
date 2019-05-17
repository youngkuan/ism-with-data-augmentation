# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import os

import torch
import torch.nn as nn

from datasets import get_loaders
from modules.discrminator import STAGE1_D, STAGE2_D, MatchDiscriminator
from modules.generator import STAGE1_G, STAGE2_G
from utils import deserialize_vocab
from utils import save_discriminator_checkpoint, weights_init, save_img_results

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')


class Trainer(object):

    def __init__(self, arguments):
        self.epochs = arguments['epochs']
        self.batch_size = arguments['batch_size']
        self.num_workers = arguments['num_workers']
        self.learning_rate = arguments['learning_rate']
        self.beta1 = arguments['beta1']
        self.kl_coef = arguments['kl_coef']
        self.margin = arguments['margin']
        self.lr_decay_step = arguments['lr_decay_step']
        self.nz = arguments['noise_dim']
        self.image_size = arguments["image_size"]
        self.gpus = arguments["gpus"]

        self.model_save_path = arguments['model_save_path']
        self.loss_save_path = arguments['loss_save_path']
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

        self.generator_data_loader, self.match_data_loader = get_loaders(arguments, vocab, self.batch_size,
                                                                         self.num_workers)

        # 图像生成器
        self.region_generator = STAGE1_G(arguments).cuda()
        self.region_generator.apply(weights_init)

        self.image_generator = STAGE2_G(arguments).cuda()
        self.image_generator.apply(weights_init)

        # 合成图像判别器
        self.region_discriminator = STAGE1_D(arguments).cuda()
        self.image_discriminator = STAGE2_D(arguments).cuda()

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

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

        self.match_discriminator_optimizer = \
            torch.optim.Adam(self.match_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.real_region_labels = torch.FloatTensor(self.batch_size, self.r).fill_(1).cuda()
        self.fake_region_labels = torch.FloatTensor(self.batch_size, self.r).fill_(0).cuda()

        self.real_image_labels = torch.FloatTensor(self.batch_size).fill_(1).cuda()
        self.fake_image_labels = torch.FloatTensor(self.batch_size).fill_(0).cuda()

    def update_stage1(self, regions, captions, hidden, lengths, criterion, epoch, iteration):
        # 更新判别器
        self.region_discriminator_optimizer.zero_grad()

        fake_regions, penal, hidden, mu = self.region_generator(captions, hidden, lengths)

        fake_regions = fake_regions.view(fake_regions.size()[0] * fake_regions.size()[1],
                                         fake_regions.size()[2],
                                         fake_regions.size()[3],
                                         fake_regions.size()[4])
        fake_region_scores = nn.parallel.data_parallel(self.region_discriminator, fake_regions, self.gpus)

        regions = regions.view(regions.size()[0] * regions.size()[1],
                               regions.size()[2],
                               regions.size()[3],
                               regions.size()[4])
        real_region_scores = nn.parallel.data_parallel(self.region_discriminator, regions, self.gpus)

        errD_fake = criterion(fake_region_scores, self.fake_region_labels)
        errD_real = criterion(real_region_scores, self.real_region_labels)
        errD = errD_real.sum() + errD_fake.sum() + penal

        errD.backward(retain_graph=True)
        self.region_discriminator_optimizer.step()

        # 更新生成器
        self.region_generator_optimizer.zero_grad()

        fake_regions, penal, hidden, mu = self.region_generator(captions, hidden, lengths)
        fake_regions = fake_regions.view(fake_regions.size()[0] * fake_regions.size()[1],
                                         fake_regions.size()[2],
                                         fake_regions.size()[3], fake_regions.size()[4])

        fake_region_scores = nn.parallel.data_parallel(self.region_discriminator, fake_regions, self.gpus)
        errG_fake = criterion(fake_region_scores, self.real_region_labels)
        errG = errG_fake.sum() + penal

        errG.backward(retain_graph=True)
        self.region_generator_optimizer.step()

        save_img_results(regions, fake_regions, epoch, self.synthetic_region_path, self.batch_size)
        print("Epoch: %d, iteration: %d, STAGE I , errD: %f, errG: %f" % (epoch, iteration, errD.data, errG.data))

    def update_stage2(self, images, captions, hidden, lengths, criterion, epoch, iteration):
        # 更新判别器
        self.image_discriminator_optimizer.zero_grad()

        fake_images, penal, hidden, mu = self.image_generator(captions, hidden, lengths)

        fake_image_scores = nn.parallel.data_parallel(self.image_discriminator, fake_images, self.gpus)
        real_image_scores = nn.parallel.data_parallel(self.image_discriminator, images, self.gpus)

        errD_fake = criterion(fake_image_scores, self.fake_image_labels)
        errD_real = criterion(real_image_scores, self.real_image_labels)
        errD = errD_real.sum() + errD_fake.sum() + penal

        errD.backward(retain_graph=True)
        self.image_discriminator_optimizer.step()

        # 更新生成器
        self.image_generator_optimizer.zero_grad()

        fake_images, penal, hidden, mu = self.image_generator(captions, hidden, lengths)

        fake_image_scores = nn.parallel.data_parallel(self.image_discriminator, fake_images, self.gpus)
        errG_fake = criterion(fake_image_scores, self.real_image_labels)
        errG = errG_fake.sum() + penal

        errG.backward(retain_graph=True)
        self.image_generator_optimizer.step()

        save_img_results(images, fake_images, epoch, self.synthetic_image_path, self.batch_size)
        print("Epoch: %d, iteration: %d, STAGE II, errD: %f, errG: %f" % (epoch, iteration, errD.data, errG.data))


    def train_generator(self):
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

            hidden = self.image_generator.txt_enc.init_hidden(self.batch_size)

            for i, train_data in enumerate(self.generator_data_loader):

                images, regions, captions, lengths, indexes = train_data
                if torch.cuda.is_available():
                    images = images.cuda()
                    regions = regions.cuda()
                    captions = captions.cuda()

                self.update_stage1(regions, captions, hidden, lengths, bce_loss, epoch, i)
                self.update_stage2(images, captions, hidden, lengths, bce_loss, epoch, i)


    def train_matching_gan(self):
        margin_ranking_loss = nn.MarginRankingLoss(self.margin)
        for epoch in range(self.epochs):
            for sample in self.match_data_loader:
                images = sample['images']
                embeddings = sample['embeddings']
                unmatched_images = sample['unmatched_images']

                images = images.cuda()
                embeddings = embeddings.cuda()
                unmatched_images = unmatched_images.cuda()

                # 更新判别器
                self.match_discriminator_optimizer.zero_grad()

                fake_images, _, _ = self.image_generator(embeddings)

                matching_scores = self.match_discriminator(images, embeddings)
                unmatched_image_scores = self.match_discriminator(unmatched_images, embeddings)
                fake_image_scores = self.match_discriminator(fake_images, embeddings)

                real_labels = torch.ones(images.size(0)).cuda()
                loss1 = margin_ranking_loss(matching_scores, unmatched_image_scores, real_labels)
                loss2 = margin_ranking_loss(fake_image_scores, unmatched_image_scores, real_labels)

                # discriminator_loss = loss1 + loss2
                discriminator_loss = loss1
                discriminator_loss.backward()

                self.match_discriminator_optimizer.step()
                print("Epoch: %d, discriminator_loss= %f" % (epoch, discriminator_loss.data))

            if (epoch + 1) == self.epochs:
                save_discriminator_checkpoint(self.match_discriminator, self.model_save_path, epoch)

    def train(self):
        self.train_generator()
        self.train_matching_gan()
