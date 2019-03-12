# -*- coding: utf-8 -*-

import torch.nn as nn

from downsample import DownsampleNetwork
from sentence_decoder import SentenceDecoder
from sentence_encoder import SentenceEncoder
from upsample import UpsampleNetwork


class SentenceGenerator(nn.Module):

    def __init__(self, arguments):
        super(SentenceGenerator, self).__init__()

        self.downsapmle_block = DownsampleNetwork(arguments)
        self.sentence_decoder_block = SentenceDecoder(arguments)

    def forward(self, image):
        image_feature = self.downsapmle_block(image)
        sentence, hidden = self.sentence_decoder_block(image_feature)

        return sentence


class ImageGenerator(nn.Module):

    def __init__(self, arguments):
        super(ImageGenerator, self).__init__()

        self.sentence_encoder_block = SentenceEncoder(arguments)
        self.upsample_block = UpsampleNetwork(arguments)


    def forward(self, sentence):
        sentence_embedding = self.sentence_encoder_block(sentence)
        image = self.upsample_block(sentence_embedding)

        return image