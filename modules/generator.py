# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from upsample import UpsampleNetwork

class ImageGenerator(nn.Module):

    def __init__(self, arguments):
        super(ImageGenerator, self).__init__()
        self.noise_dim = arguments['noise_dim']
        self.upsample_block = UpsampleNetwork(arguments)

    def forward(self, embeddings):
        noise = torch.FloatTensor(embeddings.size(0), self.noise_dim).cuda()
        noise.data.normal_(0, 1)
        image, mu, logvar = self.upsample_block(embeddings, noise)

        return image, mu, logvar
