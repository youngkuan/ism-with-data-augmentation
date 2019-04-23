# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import os

import torch
import torch.nn as nn

from datasets import get_loaders
from modules.discrminator import FakeImageDiscriminator, StackFakeImageDiscriminator, MatchDiscriminator
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
        self.fake_image_discriminator = FakeImageDiscriminator(arguments).cuda()

        self.stack_fake_image_discriminator = StackFakeImageDiscriminator(arguments).cuda()
        self.stack_fake_image_discriminator.apply(weights_init)

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

        # 优化器
        self.image_generator_optimizer = \
            torch.optim.Adam(self.image_generator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.fake_image_discriminator_optimizer = \
            torch.optim.Adam(self.fake_image_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))
        self.stack_fake_image_discriminator_optimizer = \
            torch.optim.Adam(self.stack_fake_image_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

        self.match_discriminator_optimizer = \
            torch.optim.Adam(self.match_discriminator.parameters(),
                             self.learning_rate, (self.beta1, 0.999))

    def train_generator(self):
        bce_loss = nn.BCELoss()
        losses = []
        for epoch in range(self.epochs):
            if epoch % self.lr_decay_step == 0 and epoch > 0:
                self.learning_rate *= 0.5
                for param_group in self.image_generator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in self.stack_fake_image_discriminator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            for i, train_data in enumerate(self.generator_data_loader):
                # images -> batch*36*2048
                images, captions, lengths, ids = zip(*train_data)

                # fake_images -> batch*word_count*(3*64*64)
                fake_images, cap_lens, mu, logvar = self.image_generator(captions, lengths)



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
