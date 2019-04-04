# -*- coding: utf-8 -*-

####################### 网络训练 ############################


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets import FakeImageDataLoader, MatchDataLoader
from modules.discrminator import FakeImageDiscriminator, MatchDiscriminator
from modules.generator import ImageGenerator
from utils import save_discriminator_checkpoint, save_loss, save_img_results, KL_loss

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

        # data loader
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.fake_image_data_set = FakeImageDataLoader(arguments, transform=image_transform)
        self.fake_image_data_loader = DataLoader(self.fake_image_data_set, batch_size=self.batch_size, shuffle=True,
                                                 num_workers=self.num_workers)
        self.match_data_set = MatchDataLoader(arguments)
        self.match_data_loader = DataLoader(self.match_data_set, batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.num_workers)

        # 图像生成器
        self.image_generator = ImageGenerator(arguments).cuda(device=cuda0)

        # 合成图像判别器
        self.fake_image_discriminator = FakeImageDiscriminator(arguments).cuda(device=cuda0)

        # 图像文本匹配判别器
        self.match_discriminator = MatchDiscriminator(arguments).cuda()

        # 优化器
        self.image_generator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.image_generator.parameters())
            , self.learning_rate, (self.beta1, 0.999))

        self.fake_image_discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.fake_image_discriminator.parameters())
            , self.learning_rate, (self.beta1, 0.999))
        self.match_discriminator_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.match_discriminator.parameters())
            , self.learning_rate, (self.beta1, 0.999))

    def train_fake_image_gan(self):
        bce_loss = nn.BCELoss()
        losses = []
        for epoch in range(self.epochs):
            if epoch % self.lr_decay_step == 0 and epoch > 0:
                self.learning_rate *= 0.5
                for param_group in self.image_generator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in self.fake_image_discriminator_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            for sample in self.fake_image_data_loader:
                embeddings = sample['embeddings']
                matched_images_cpu = sample['images']
                unmatched_images_cpu = sample['unmatched_images']

                embeddings = embeddings.cuda(device=cuda0)
                matched_images = matched_images_cpu.cuda(device=cuda0)
                unmatched_images = unmatched_images_cpu.cuda(device=cuda0)

                real_labels = torch.ones(matched_images.size(0)).cuda(device=cuda0)
                fake_labels = torch.zeros(matched_images.size(0)).cuda(device=cuda0)

                # 更新判别器
                self.fake_image_discriminator.zero_grad()
                # 计算损失
                fake_images, mu, logvar = self.image_generator(embeddings)

                matched_scores = self.fake_image_discriminator(matched_images, embeddings)
                unmatched_scores = self.fake_image_discriminator(unmatched_images, embeddings)
                fake_scores = self.fake_image_discriminator(fake_images, embeddings)

                matched_loss = bce_loss(matched_scores, real_labels)
                unmatched_loss = bce_loss(unmatched_scores, fake_labels)
                fake_loss = bce_loss(fake_scores, fake_labels)

                discriminator_loss = matched_loss + (unmatched_loss + fake_loss) * 0.5
                # 损失反向传递
                discriminator_loss.backward()
                self.fake_image_discriminator_optimizer.step()

                # 更新生成器
                self.image_generator.zero_grad()
                fake_images, mu, logvar = self.image_generator(embeddings)
                fake_scores = self.fake_image_discriminator(fake_images, embeddings)

                fake_loss = bce_loss(fake_scores, real_labels)
                kl_loss = KL_loss(mu, logvar)
                generator_loss = fake_loss + self.kl_coef * kl_loss
                generator_loss.backward()

                self.image_generator_optimizer.step()
                print("Epoch: %d, generator_loss= %f, discriminator_loss= %f" %
                      (epoch, generator_loss.data, discriminator_loss.data))
                save_img_results(matched_images_cpu, fake_images, epoch, self.synthetic_image_path, self.batch_size)
            loss = [epoch, generator_loss.data, discriminator_loss.data,
                    matched_loss.data, unmatched_loss.data, fake_loss.data]
            losses.append(loss)
        save_loss(losses, self.loss_save_path)

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

                fake_images = self.image_generator(embeddings)

                matching_scores = self.match_discriminator(images, embeddings)
                unmatched_image_scores = self.match_discriminator(unmatched_images, embeddings)
                fake_image_scores = self.match_discriminator(fake_images, embeddings)

                real_labels = torch.ones(images.size(0)).cuda()
                loss1 = margin_ranking_loss(matching_scores, unmatched_image_scores, real_labels)
                loss2 = margin_ranking_loss(fake_image_scores, unmatched_image_scores, real_labels)

                discriminator_loss = loss1 + loss2
                discriminator_loss.backward()

                self.match_discriminator_optimizer.step()
                print("Epoch: %d, discriminator_loss= %f" % (epoch, discriminator_loss.data))

            if (epoch + 1) == self.epochs:
                save_discriminator_checkpoint(self.match_discriminator, self.model_save_path, epoch)

    def train(self):
        self.train_fake_image_gan()
        self.train_matching_gan()
