# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import FakeImageDataLoader,FakeSentenceDataLoader,MatchDataLoader
from loss import RankingLoss, FakeImageLoss, FakeSentenceLoss
from modules.discrminator import FakeImageDiscriminator, FakeSentenceDiscriminator, MatchDiscriminator
from modules.generator import ImageGenerator, SentenceGenerator


class Trainer(object):

    def __init__(self, arguments):

        self.epochs = arguments['epochs']
        self.batch_size = arguments['batch_size']
        self.num_workers = arguments['num_workers']
        self.learning_rate = arguments['learning_rate']
        self.beta1 = arguments['beta1']
        self.l1_coef = arguments['l1_coef']
        self.margin = arguments['margin']

        # data loader
        self.fake_image_data_set = FakeImageDataLoader(arguments)
        self.fake_image_data_loader = DataLoader(self.fake_image_data_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers)

        self.fake_sentence_data_set = FakeSentenceDataLoader(arguments)
        self.fake_sentence_data_loader = DataLoader(self.fake_sentence_data_set, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        self.match_data_set = MatchDataLoader(arguments)
        self.match_data_loader = DataLoader(self.match_data_set, batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers)

        # 图像生成器
        self.image_generator = ImageGenerator(arguments).cuda()
        # 文本生成器
        self.sentence_generator = SentenceGenerator(arguments).cuda()

        # 合成图像判别器
        self.fake_image_discriminator = FakeImageDiscriminator(arguments).cuda()

        # 合成文本判别器
        self.fake_sentence_discriminator = FakeSentenceDiscriminator(arguments).cuda()

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

        # 损失函数
        self.ranking_loss = RankingLoss(arguments)
        self.fake_image_loss = FakeImageLoss(arguments)
        self.fake_sentence_loss = FakeSentenceLoss(arguments)

        # 优化器
        self.image_generator_optimizer = torch.optim.Adam(self.image_generator.parameters()
                                                          , self.learning_rate, (self.beta1, 0.999))
        self.sentence_generator_optimizer = torch.optim.Adam(self.sentence_generator.parameters()
                                                             , self.learning_rate, (self.beta1, 0.999))

        self.fake_image_discriminator_optimizer = torch.optim.Adam(self.fake_image_discriminator.parameters()
                                                                   , self.learning_rate, (self.beta1, 0.999))
        self.fake_sentence_discriminator_optimizer = torch.optim.Adam(self.fake_sentence_discriminator.parameters()
                                                                      , self.learning_rate, (self.beta1, 0.999))
        self.match_discriminator_optimizer = torch.optim.Adam(self.match_discriminator.parameters()
                                                              , self.learning_rate, (self.beta1, 0.999))

    def train_fake_image_gan(self):
        bce_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()

        for epoch in range(self.epochs):
            for sample in self.fake_image_data_loader:
                sentences = sample['sentences']
                matched_images = sample['matched_images']
                unmatched_images = sample['unmatched_images']

                sentences = torch.tensor(sentences, requires_grad=False).cuda()
                matched_images = torch.tensor(matched_images, requires_grad=False).cuda()
                unmatched_images = torch.tensor(unmatched_images, requires_grad=False).cuda()

                real_labels = torch.ones(matched_images.size(0)).cuda()
                fake_labels = torch.zeros(matched_images.size(0)).cuda()

                # 更新判别器
                self.fake_image_discriminator_optimizer.zero_grad()
                # 计算损失
                fake_images = self.image_generator(sentences)

                matched_scores = self.fake_image_discriminator(matched_images, sentences)
                unmatched_scores = self.fake_image_discriminator(unmatched_images, sentences)
                fake_scores = self.fake_image_discriminator(fake_images, sentences)

                matched_loss = bce_loss(matched_scores, real_labels)
                unmatched_loss = bce_loss(unmatched_scores, fake_labels)
                fake_loss = bce_loss(fake_scores, fake_labels)

                discriminator_loss = matched_loss + unmatched_loss + fake_loss
                # 损失反向传递
                discriminator_loss.backward()
                self.fake_image_discriminator_optimizer.step()

                # 更新生成器
                self.image_generator_optimizer.zero_grad()

                fake_images = self.image_generator(sentences)
                fake_scores = self.fake_image_discriminator(fake_images, sentences)

                fake_loss = bce_loss(fake_scores, real_labels)

                generator_loss = fake_loss + self.l1_coef * l1_loss(fake_images, matched_images)
                generator_loss.backward()

                self.image_generator_optimizer.step()

    def train_fake_sentence_gan(self):
        bce_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()

        for epoch in range(self.epochs):
            for sample in self.fake_sentence_data_loader:
                images = sample['images']
                matched_sentences = sample['matched_sentences']
                unmatched_sentences = sample['unmatched_sentences']

                images = torch.tensor(images, requires_grad=False).cuda()
                matched_sentences = torch.tensor(matched_sentences, requires_grad=False).cuda()
                unmatched_sentences = torch.tensor(unmatched_sentences, requires_grad=False).cuda()

                real_labels = torch.ones(images.size(0)).cuda()
                fake_labels = torch.zeros(images.size(0)).cuda()

                # 更新判别器
                self.fake_sentence_discriminator_optimizer.zero_grad()
                # 计算损失
                # lengths = [len(matched_sentence) for matched_sentence in matched_sentences]
                fake_sentences = self.sentence_generator(images)

                matched_scores = self.fake_sentence_discriminator(images, matched_sentences)
                unmatched_scores = self.fake_sentence_discriminator(images, unmatched_sentences)
                fake_scores = self.fake_sentence_discriminator(images, fake_sentences)

                matched_loss = bce_loss(matched_scores, real_labels)
                unmatched_loss = bce_loss(unmatched_scores, fake_labels)
                fake_loss = bce_loss(fake_scores, fake_labels)

                discriminator_loss = matched_loss + unmatched_loss + fake_loss
                # 损失反向传递
                discriminator_loss.backward()
                self.fake_sentence_discriminator_optimizer.step()

                # 更新生成器
                self.sentence_generator_optimizer.zero_grad()

                fake_sentences = self.sentence_generator(images)
                fake_scores = self.fake_sentence_discriminator(images, fake_sentences)

                fake_loss = bce_loss(fake_scores, real_labels)

                # generator_loss = fake_loss + self.l1_coef * l1_loss(fake_sentences, matched_sentences)
                generator_loss = fake_loss
                generator_loss.backward()

                self.sentence_generator_optimizer.step()

    def train_matching_gan(self):
        margin_ranking_loss = nn.MarginRankingLoss(self.margin)

        for epoch in range(self.epochs):
            for sample in self.match_data_loader:
                images = sample['images']
                sentences = sample['sentences']
                unmatched_images = sample['unmatched_images']
                unmatched_sentences = sample['unmatched_sentences']

                images = torch.tensor(images, requires_grad=False).cuda()
                sentences = torch.tensor(sentences, requires_grad=False).cuda()
                unmatched_images = torch.tensor(unmatched_images, requires_grad=False).cuda()
                unmatched_sentences = torch.tensor(unmatched_sentences, requires_grad=False).cuda()

                # 更新判别器
                self.match_discriminator_optimizer.zero_grad()

                fake_images = self.image_generator(sentences)
                fake_sentences = self.sentence_generator(images)

                matching_scores = self.match_discriminator(images, sentences)
                unmatched_sentence_scores = self.match_discriminator(images, unmatched_sentences)
                unmatched_image_scores = self.match_discriminator(unmatched_images, sentences)
                fake_image_scores = self.match_discriminator(fake_images, sentences)
                fake_sentence_scores = self.match_discriminator(images, fake_sentences)

                real_labels = torch.ones(images.size(0)).cuda()
                loss1 = margin_ranking_loss(matching_scores, unmatched_sentence_scores, real_labels)
                loss2 = margin_ranking_loss(matching_scores, unmatched_image_scores, real_labels)
                loss3 = margin_ranking_loss(fake_image_scores, unmatched_image_scores, real_labels)
                loss4 = margin_ranking_loss(fake_sentence_scores, unmatched_sentence_scores, real_labels)

                discriminator_loss = loss1 + loss2 + loss3 + loss4
                discriminator_loss.backward()

                self.match_discriminator_optimizer.step()

    def train(self):
        # self.train_fake_image_gan()
        self.train_fake_sentence_gan()
        self.train_matching_gan()
