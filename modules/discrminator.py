# -*- coding: utf-8 -*-

####################### 判别器网络 ############################
# 三个判别器，分别是：
# 1. 合成图像判别器
# 1. 合成文本判别器
# 1. 图像文本匹配判别器
##### reference


import torch
import torch.nn as nn

from downsample import DownsampleNetwork


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FakeImageDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(FakeImageDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]
        self.project_size = arguments["project_size"]

        self.downsample_block = DownsampleNetwork(arguments)

        self.sentence_projector = nn.Sequential(
            nn.Linear(in_features=self.sentence_embedding_size, out_features=self.project_size),
            nn.BatchNorm1d(num_features=self.project_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.project_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        # weight init, inspired by tutorial
        self.sentence_projector.apply(self.weights_init)
        self.net.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, images, embeddings):
        image_features = self.downsample_block(images)
        project_embedding = self.sentence_projector(embeddings)

        concat_feature = torch.cat([image_features, project_embedding], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)
        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)


class MatchDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(MatchDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]
        self.project_size = arguments["project_size"]

        self.downsample_block = DownsampleNetwork(arguments)

        self.sentence_projector = nn.Sequential(
            nn.Linear(in_features=self.sentence_embedding_size, out_features=self.project_size),
            nn.BatchNorm1d(num_features=self.project_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.project_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.init_weights()

    def init_weights(self):
        # weight init, inspired by tutorial
        self.sentence_projector.apply(self.weights_init)
        self.net.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, image, embeddings):
        image_feature = self.downsample_block(image)
        project_embedding = self.sentence_projector(embeddings)

        concat_feature = torch.cat([image_feature, project_embedding], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)

        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)


class StackFakeImageDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(StackFakeImageDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.ndf = arguments['ndf']
        self.nef = arguments["condition_dimension"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]
        self.project_size = arguments["project_size"]
        ndf, nef = self.ndf, self.nef

        # input image size 3*64*64
        self.encode_image = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.outlogits = nn.Sequential(
            conv3x3(ndf * 8 + nef, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, image, condition):
        image_feature = self.encode_image(image)

        condition = condition.view(-1, self.nef, 1, 1)
        condition = condition.repeat(1, 1, 4, 4)
        # state size (ngf+nef) x 4 x 4
        code = torch.cat((image_feature, condition), 1)

        output = self.outlogits(code)
        return output.view(-1)
