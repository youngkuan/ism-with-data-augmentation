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
from sentence_encoder import SentenceEncoder


class FakeImageDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(FakeImageDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]

        self.downsample_block = DownsampleNetwork(arguments)
        self.sentence_encoder_block = SentenceEncoder(arguments)

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.sentence_embedding_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.encoder_hidden = self.sentence_encoder_block.initHidden()

    def forward(self, image, sentence):
        image_feature = self.downsample_block(image)
        sentence_embedding = self.sentence_encoder_block(sentence, self.encoder_hidden)

        concat_feature = torch.cat([image_feature, sentence_embedding], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)

        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)


class FakeSentenceDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(FakeSentenceDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]

        self.downsample_block = DownsampleNetwork(arguments)
        self.sentence_encoder_block = SentenceEncoder(arguments)

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.sentence_embedding_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.encoder_hidden = self.sentence_encoder_block.initHidden()

    def forward(self, image, sentence):
        image_feature = self.downsample_block(image)
        sentence_embedding = self.sentence_encoder_block(sentence, self.encoder_hidden)
        concat_feature = torch.cat([image_feature, sentence_embedding], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)
        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)


class MatchDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(MatchDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]

        self.downsample_block = DownsampleNetwork(arguments)
        self.sentence_encoder_block = SentenceEncoder(arguments)

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.sentence_embedding_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.encoder_hidden = self.sentence_encoder_block.initHidden()
        self.encoder_hidden = None

    def forward(self, image, sentence):
        image_feature = self.downsample_block(image)
        sentence_embedding = self.sentence_encoder_block(sentence, self.encoder_hidden)

        concat_feature = torch.cat([image_feature, sentence_embedding], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)

        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)
