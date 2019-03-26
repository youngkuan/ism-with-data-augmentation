# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from datasets import FakeImageDataLoader, FakeSentenceDataLoader, MatchDataLoader
from datasets import load_validation_set, collate_fn
from evaluation import i2t, t2i
from loss import RankingLoss, FakeImageLoss, FakeSentenceLoss
from modules.discrminator import FakeImageDiscriminator, FakeSentenceDiscriminator, MatchDiscriminator
from modules.downsample import DownsampleNetwork
from modules.generator import ImageGenerator
from modules.sentence_decoder import SentenceDecoder
from utils import save_image, save_sentence, convert_indexes2sentence,save_discriminator_checkpoint

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')


class Trainer(object):

    def __init__(self, arguments):
        self.epochs = arguments['epochs']
        self.batch_size = arguments['batch_size']
        self.num_workers = arguments['num_workers']
        self.learning_rate = arguments['learning_rate']
        self.beta1 = arguments['beta1']
        self.l1_coef = arguments['l1_coef']
        self.margin = arguments['margin']
        self.model_save_path = arguments['model_save_path']
        self.arguments = arguments

        # data loader
        self.fake_image_data_set = FakeImageDataLoader(arguments)
        self.fake_image_data_loader = DataLoader(self.fake_image_data_set, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)

        self.fake_sentence_data_set = FakeSentenceDataLoader(arguments)
        self.fake_sentence_data_loader = DataLoader(self.fake_sentence_data_set, batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers, collate_fn=collate_fn)

        self.match_data_set = MatchDataLoader(arguments)
        self.match_data_loader = DataLoader(self.match_data_set, batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.num_workers)

        # 图像生成器
        self.image_generator = ImageGenerator(arguments).cuda(device=cuda0)
        # 文本生成器
        # self.sentence_generator = SentenceGenerator(arguments).cuda(device=cuda1)

        self.downsapmle_block = DownsampleNetwork(arguments).cuda()
        self.sentence_decoder_block = SentenceDecoder(arguments).cuda()

        # 合成图像判别器
        self.fake_image_discriminator = FakeImageDiscriminator(arguments).cuda(device=cuda0)

        # 合成文本判别器
        self.fake_sentence_discriminator = FakeSentenceDiscriminator(arguments).cuda(device=cuda1)

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

        # 损失函数
        self.ranking_loss = RankingLoss(arguments)
        self.fake_image_loss = FakeImageLoss(arguments)
        self.fake_sentence_loss = FakeSentenceLoss(arguments)

        # 优化器
        self.image_generator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.image_generator.parameters())
            , self.learning_rate, (self.beta1, 0.999))
        # self.sentence_generator_optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.sentence_generator.parameters())
        #     , self.learning_rate, (self.beta1, 0.999))
        self.downsapmle_block_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.downsapmle_block.parameters())
            , self.learning_rate, (self.beta1, 0.999))
        self.sentence_decoder_block_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.sentence_decoder_block.parameters())
            , self.learning_rate, (self.beta1, 0.999))

        self.fake_image_discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.fake_image_discriminator.parameters())
            , self.learning_rate, (self.beta1, 0.999))
        self.fake_sentence_discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.fake_sentence_discriminator.parameters())
            , self.learning_rate, (self.beta1, 0.999))
        self.match_discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.match_discriminator.parameters())
            , self.learning_rate, (self.beta1, 0.999))

    def train_fake_image_gan(self):
        bce_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()

        for epoch in range(self.epochs):
            for sample in self.fake_image_data_loader:
                sentences = sample['sentences']
                matched_images = sample['matched_images']
                unmatched_images = sample['unmatched_images']

                sentences = torch.tensor(sentences, requires_grad=False).cuda(device=cuda0)
                matched_images = torch.tensor(matched_images, requires_grad=False).cuda(device=cuda0)
                unmatched_images = torch.tensor(unmatched_images, requires_grad=False).cuda(device=cuda0)

                real_labels = torch.ones(matched_images.size(0)).cuda(device=cuda0)
                fake_labels = torch.zeros(matched_images.size(0)).cuda(device=cuda0)

                # 更新判别器
                for param in self.image_generator.parameters():
                    param.requires_grad = False
                for param in self.fake_image_discriminator.parameters():
                    param.requires_grad = True
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
                for param in self.image_generator.parameters():
                    param.requires_grad = True
                for param in self.fake_image_discriminator.parameters():
                    param.requires_grad = False

                self.image_generator_optimizer.zero_grad()

                fake_images = self.image_generator(sentences)
                fake_scores = self.fake_image_discriminator(fake_images, sentences)

                fake_loss = bce_loss(fake_scores, real_labels)

                generator_loss = fake_loss + self.l1_coef * l1_loss(fake_images, matched_images)
                generator_loss.backward()

                self.image_generator_optimizer.step()
                print("Epoch: %d, generator_loss= %f, discriminator_loss= %f" %
                      (epoch, generator_loss.data, discriminator_loss.data))

            val_images, val_sentences = load_validation_set(self.arguments)
            val_sentences = torch.tensor(val_sentences[:10], requires_grad=False).cuda()
            val_images = torch.tensor(val_images[:10], requires_grad=False).cuda()
            fake_images = self.image_generator(val_sentences)
            index = 0
            for val_image, fake_image in zip(val_images, fake_images):
                save_image(val_image, self.arguments['synthetic_image_path'], "val_" + str(index) + ".jpg")
                save_image(fake_image, self.arguments['synthetic_image_path'], "fake_" + str(index) + ".jpg")
                index = index + 1

    def train_fake_sentence_gan(self):
        bce_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()

        for epoch in range(self.epochs):
            for sample in self.fake_sentence_data_loader:
                images = sample['images']
                matched_sentences = sample['matched_sentences']
                unmatched_sentences = sample['unmatched_sentences']

                images = torch.tensor(images, requires_grad=False).cuda(device=cuda1)
                matched_sentences = torch.tensor(matched_sentences, requires_grad=False).cuda(device=cuda1)
                unmatched_sentences = torch.tensor(unmatched_sentences, requires_grad=False).cuda(device=cuda1)

                real_labels = torch.ones(images.size(0)).cuda(device=cuda1)
                fake_labels = torch.zeros(images.size(0)).cuda(device=cuda1)

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
                print("Epoch: %d, generator_loss= %f, discriminator_loss= %f" %
                      (epoch, generator_loss.data, discriminator_loss.data))

            val_images, val_sentences = load_validation_set(self.arguments)
            val_sentences = torch.tensor(val_sentences[:10], requires_grad=False).cuda()
            val_images = torch.tensor(val_images[:10], requires_grad=False).cuda()

            fake_sentences = self.sentence_generator(val_images)
            fake_sentences = fake_sentences.cpu().numpy()
            fake_sentences = convert_indexes2sentence(self.arguments['idx2word'], fake_sentences)
            save_sentence(val_sentences, fake_sentences, self.arguments['synthetic_sentence_path'])

    def train_fake_sentence_generator(self):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            for step, (images, captions, lengths) in enumerate(self.fake_sentence_data_loader):
                images = torch.tensor(images, requires_grad=False).cuda(device=cuda1)
                captions = torch.tensor(captions, requires_grad=False).cuda(device=cuda1)

                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # 更新生成器
                # self.sentence_generator_optimizer.zero_grad()
                self.downsapmle_block_optimizer.zero_grad()
                self.sentence_decoder_block_optimizer.zero_grad()

                image_features = self.downsapmle_block(images)
                fake_sentences_distribution = self.sentence_decoder_block(image_features, captions, lengths)

                generator_loss = criterion(fake_sentences_distribution, targets)

                generator_loss.backward()

                # self.sentence_generator_optimizer.step()
                self.downsapmle_block_optimizer.step()
                self.sentence_decoder_block_optimizer.step()
                print("Epoch: %d, generator_loss = %5.4f" % (epoch, generator_loss.item()))

            val_images, val_sentences = load_validation_set(self.arguments)
            val_sentences = torch.tensor(val_sentences[:10], requires_grad=False).cuda()
            val_images = torch.tensor(val_images[:10], requires_grad=False).cuda()

            val_image_features = self.downsapmle_block(val_images)
            fake_sentences = self.sentence_decoder_block.sample(val_image_features)
            fake_sentences = convert_indexes2sentence(self.arguments['idx2word'], fake_sentences)
            save_sentence(val_sentences, fake_sentences, self.arguments['synthetic_sentence_path'])

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
                if self.arguments['use_sentence_generator']:
                    # fake_sentences = self.sentence_generator(images)
                    with torch.no_grad():
                        image_features = self.downsapmle_block(images)
                        fake_sentences = self.sentence_decoder_block.sample(image_features)

                        fake_sentences = convert_indexes2sentence(self.arguments['idx2word'], fake_sentences)
                        xs = []
                        for sentence in fake_sentences:
                            sentence = [self.arguments['word2idx'][w] if self.arguments['word2idx'][w] < self.arguments[
                                'word_number'] else 1 for w in
                                        sentence.split()]
                            x = np.zeros(self.arguments['sentence_max_length']).astype('int64')
                            if len(sentence) < self.arguments['sentence_max_length']:
                                x[:len(sentence)] = sentence
                            else:
                                x[:] = sentence[:self.arguments['sentence_max_length']]
                            xs.append(x)
                        fake_sentences = np.stack(xs, 0)
                        fake_sentences = torch.LongTensor(fake_sentences)
                        fake_sentences = torch.tensor(fake_sentences, requires_grad=False).cuda()
                        fake_sentence_scores = self.match_discriminator(images, fake_sentences)
                        loss4 = margin_ranking_loss(fake_sentence_scores, unmatched_sentence_scores, real_labels)

                matching_scores = self.match_discriminator(images, sentences)
                unmatched_sentence_scores = self.match_discriminator(images, unmatched_sentences)
                unmatched_image_scores = self.match_discriminator(unmatched_images, sentences)
                fake_image_scores = self.match_discriminator(fake_images, sentences)

                real_labels = torch.ones(images.size(0)).cuda()
                loss1 = margin_ranking_loss(matching_scores, unmatched_sentence_scores, real_labels)
                loss2 = margin_ranking_loss(matching_scores, unmatched_image_scores, real_labels)
                loss3 = margin_ranking_loss(fake_image_scores, unmatched_image_scores, real_labels)

                if self.arguments['use_sentence_generator']:
                    discriminator_loss = loss1 + loss2 + loss3 + loss4
                else:
                    discriminator_loss = loss1 + loss2 + loss3

                discriminator_loss.backward()

                self.match_discriminator_optimizer.step()
                print("Epoch: %d, discriminator_loss= %f" % (epoch, discriminator_loss.data))

            if (epoch + 1) == self.epochs:
                save_discriminator_checkpoint(self.match_discriminator,self.model_save_path,epoch)

            val_images, val_sentences = load_validation_set(self.arguments)
            val_sentences = torch.tensor(val_sentences, requires_grad=False).cuda()
            val_images = torch.tensor(val_images, requires_grad=False).cuda()
            i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(self.match_discriminator, val_images, val_sentences)
            t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(self.match_discriminator, val_sentences, val_images)
            print "Image to Text: %.2f, %.2f, %.2f, %.2f" \
                  % (i2t_r1, i2t_r5, i2t_r10, i2t_medr)
            print "Text to Image: %.2f, %.2f, %.2f, %.2f" \
                  % (t2i_r1, t2i_r5, t2i_r10, t2i_medr)

    def train(self):
        self.train_fake_image_gan()
        # self.train_fake_sentence_gan()
        if self.arguments['use_sentence_generator']:
            self.train_fake_sentence_generator()
        self.train_matching_gan()
