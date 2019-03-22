# -*- coding: utf-8 -*-

import torch
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

    def forward(self, images, sentences, lengths):
        image_features = self.downsapmle_block(images)

        outputs = self.sentence_decoder_block(image_features, sentences, lengths)

        return outputs

    def generate_features(self,images):

        features = self.downsapmle_block(images)

        return features


class ImageGenerator(nn.Module):

    def __init__(self, arguments):
        super(ImageGenerator, self).__init__()
        self.noise_dim = arguments['noise_dim']
        self.sentence_encoder_block = SentenceEncoder(arguments)
        # self.encoder_hidden = self.sentence_encoder_block.initHidden()
        self.encoder_hidden = None
        self.upsample_block = UpsampleNetwork(arguments)

    def forward(self, sentence):
        sentence_embedding = self.sentence_encoder_block(sentence, self.encoder_hidden)
        noise = torch.randn(sentence_embedding.size(0), self.noise_dim).cuda()

        # size: batch_size x sentence_embedding
        sentence_embedding = sentence_embedding.unsqueeze(2).unsqueeze(3)
        # size: batch_size x sentence_embedding x 1 x 1

        noise = noise.unsqueeze(2).unsqueeze(3)
        concat_embedding = torch.cat([sentence_embedding, noise], 1)

        image = self.upsample_block(concat_embedding)

        return image
