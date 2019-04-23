# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from upsample import UpsampleNetwork
from model import EncoderText

class ImageGenerator(nn.Module):

    def __init__(self, arguments):
        super(ImageGenerator, self).__init__()
        self.noise_dim = arguments['noise_dim']

        self.txt_enc = EncoderText(arguments.vocab_size, arguments.word_dim,
                                   arguments.embed_size, arguments.num_layers,
                                   use_bi_gru=arguments.bi_gru,
                                   no_txtnorm=arguments.no_txtnorm)

        self.upsample_block = UpsampleNetwork(arguments)



    def forward(self, captions, lengths):
        cap_emb, cap_lens = self.txt_enc(captions, lengths)

        noise = torch.FloatTensor(cap_emb.size(0), cap_emb.size(1),self.noise_dim).cuda()
        noise.data.normal_(0, 1)
        image, mu, logvar = self.upsample_block(cap_emb, noise)

        return image, cap_lens, mu, logvar
