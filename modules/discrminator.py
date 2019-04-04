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


class FakeImageDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(FakeImageDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]

        self.downsample_block = DownsampleNetwork(arguments)

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.sentence_embedding_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, embeddings):
        image_features = self.downsample_block(images)

        concat_feature = torch.cat([image_features, embeddings], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)
        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)


class MatchDiscriminator(nn.Module):
    def __init__(self, arguments):
        super(MatchDiscriminator, self).__init__()
        self.image_feature_size = arguments["image_feature_size"]
        self.sentence_embedding_size = arguments["sentence_embedding_size"]

        self.downsample_block = DownsampleNetwork(arguments)

        self.net = nn.Sequential(
            nn.Conv2d(self.image_feature_size + self.sentence_embedding_size, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image, embeddings):
        image_feature = self.downsample_block(image)

        concat_feature = torch.cat([image_feature, embeddings], 1)
        concat_feature = concat_feature.view(concat_feature.size()[0], concat_feature.size()[1], 1, 1)

        output = self.net(concat_feature)
        return output.view(-1, 1).squeeze(1)
