# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model import SelfAttentive
from upsample import UpsampleNetwork


class ImageGenerator(nn.Module):

    def __init__(self, arg):
        super(ImageGenerator, self).__init__()
        self.noise_dim = arg['noise_dim']
        vocab_size = arg['vocab_size']
        word_dim = arg['word_dim']
        embed_size = arg['embed_size']
        num_layers = arg['num_layers']
        da = arg['da']
        self.r = arg['r']
        emb_matrix = arg['emb_matrix']
        cuda = arg['cuda']
        use_bi_gru = arg['bi_gru']


        self.txt_enc = SelfAttentive(vocab_size, word_dim, embed_size, num_layers,
                                     da, self.r, emb_matrix, cuda, use_bi_gru)

        self.upsample_block = UpsampleNetwork(arg)

    def forward(self, captions, hidden, lengths):
        print "hidden.size(): ",hidden.size()
        # cap_emb -> batch_size*word_count*embed_size
        BM, hidden, penal, weights = self.txt_enc(captions, hidden, lengths)
        print "BM: ", BM.size()

        embed = BM.view(captions.size()[0], self.r, -1)
        noise = torch.FloatTensor(embed.size(0), embed.size(1), self.noise_dim).cuda()
        noise.data.normal_(0, 1)
        images, mus, logvars = self.upsample_block(embed, noise)

        return images, penal, hidden, mus
