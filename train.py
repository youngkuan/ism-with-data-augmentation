# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import os

import torch
import torch.nn as nn

from datasets import get_loaders
from modules.discrminator import ImageDiscriminator, MatchDiscriminator
from modules.generator import ImageGenerator
from modules.utils import deserialize_vocab
from modules.utils import save_discriminator_checkpoint, weights_init

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

        self.model_save_path = arguments['model_save_path']
        self.loss_save_path = arguments['loss_save_path']
        self.synthetic_image_path = arguments['synthetic_image_path']
        self.arguments = arguments

        self.data_dir = arguments.data_dir
        self.voc_file = arguments.voc_file
        self.train_path = os.path.join(self.data_dir, "train")
        # Load Vocabulary Wrapper
        vocab = deserialize_vocab(self.train_path, self.voc_file)

        self.generator_data_loader, self.match_data_loader = get_loaders(arguments, vocab)

        # 图像生成器
        self.image_generator = ImageGenerator(arguments).cuda()
        self.image_generator.apply(weights_init)

        # 合成图像判别器
        self.image_discriminator = ImageDiscriminator(arguments).cuda()

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

        # 优化器
        self.image_generator_optimizer = \
            torch.optim.Adam(self.image_generator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.image_discriminator_optimizer = \
            torch.optim.Adam(self.image_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.match_discriminator_optimizer = \
            torch.optim.Adam(self.match_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

    def train_generator(self):
        bce_loss = nn.BCELoss()
        real_labels = torch.FloatTensor(self.batch_size).fill_(1).cuda()
        fake_labels = torch.FloatTensor(self.batch_size).fill_(0).cuda()

        for epoch in range(self.epochs):
            if epoch % self.lr_decay_step == 0 and epoch > 0:
                self.learning_rate *= 0.5
                for param_group in self.image_generator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in self.stack_fake_image_discriminator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            for i, train_data in enumerate(self.generator_data_loader):
                # image_features -> batch*36*2048
                # images -> batch*(3*64*64)
                images, image_features, captions, lengths, indexes = zip(*train_data)

                # 更新判别器
                self.image_discriminator_optimizer.zero_grad()

                # fake_images -> batch*word_count*(3*64*64)
                # mu -> batch * condition_dimension
                fake_images, cap_lens, mu, logvar = self.image_generator(captions, lengths)

                repeated_mu = mu.view(-1, 1, mu.size()[1])
                repeated_mu = repeated_mu.repeat(fake_images.size()[0], fake_images.size()[1], mu.size()[1])
                # fake_image_scores -> batch * word_count
                fake_image_scores = self.image_discriminator(fake_images, repeated_mu)
                # get the max scores of every word count
                fake_image_scores = torch.max(fake_image_scores, dim=1)

                real_image_scores = self.image_discriminator(images, mu)
                wrong_images = images[:(images.size()[0] - 2)]
                wrong_image_scores = self.image_discriminator(wrong_images, mu)

                errD_fake = bce_loss(fake_image_scores, fake_labels)
                errD_wrong = bce_loss(wrong_image_scores, fake_labels)
                errD_real = bce_loss(real_image_scores, real_labels)

                errD = errD_real + (errD_fake + errD_wrong) * 0.5

                errD.backward()
                self.image_discriminator_optimizer.step()

                self.image_generator_optimizer.zero_grad()

                # fake_images -> batch*word_count*(3*64*64)
                # mu -> batch * condition_dimension
                fake_images, cap_lens, mu, logvar = self.image_generator(captions, lengths)

                repeated_mu = mu.view(-1, 1, mu.size()[1])
                repeated_mu = repeated_mu.repeat(fake_images.size()[0], fake_images.size()[1], mu.size()[1])
                # fake_image_scores -> batch * word_count
                fake_image_scores = self.image_discriminator(fake_images, repeated_mu)
                # get the max scores of every word count
                fake_image_scores = torch.max(fake_image_scores, dim=1)


                errD_fake = bce_loss(fake_image_scores, real_labels)

                errD_fake.backward()


                self.image_generator_optimizer.step()

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
