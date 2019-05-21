# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model import SelfAttentive
from upsample import UPSAMPLE_BLOCK


def conv3x3(in_channels, out_channels):
    """
    3x3 convolution with padding
    :param in_channels:
    :param out_channels:
    :return:
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)


class STAGE1_G(nn.Module):

    def __init__(self, arg):
        super(STAGE1_G, self).__init__()
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

        self.ngf = arg['ngf']
        self.batch_size = arg['batch_size']

        self.txt_enc = SelfAttentive(vocab_size, word_dim, embed_size, num_layers,
                                     da, self.r, emb_matrix, cuda, use_bi_gru)

        self.upsample_block = UPSAMPLE_BLOCK(arg)

        self.region = nn.Sequential(
            conv3x3(self.ngf // 16, 3),
            nn.Tanh())

    def forward(self, captions, lengths, hidden=None):
        # cap_emb -> batch_size*word_count*embed_size
        BM, hidden, penal, weights = self.txt_enc(captions, lengths, hidden)

        embed = BM.view(captions.size()[0], self.r, -1)
        noise = torch.FloatTensor(embed.size(0), embed.size(1), self.noise_dim).cuda()
        noise.data.normal_(0, 1)
        stage2_outputs, stage1_outputs, mus, logvars = self.upsample_block(embed, noise)

        stage1_outputs = self.region(stage1_outputs)
        regions = stage1_outputs.view(self.batch_size, -1,
                                      stage1_outputs.size()[1], stage1_outputs.size()[2],
                                      stage1_outputs.size()[3])

        return regions, penal, hidden, mus, BM


class STAGE2_G(nn.Module):

    def __init__(self, arg):
        super(STAGE2_G, self).__init__()
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

        self.ngf = arg['ngf']

        self.txt_enc = SelfAttentive(vocab_size, word_dim, embed_size, num_layers,
                                     da, self.r, emb_matrix, cuda, use_bi_gru)

        self.upsample_block = UPSAMPLE_BLOCK(arg)

        self.image = nn.Sequential(
            conv3x3(self.r * self.ngf // 64, 3),
            nn.Tanh())

    def forward(self, captions, lengths, hidden=None):
        # cap_emb -> batch_size*word_count*embed_size
        BM, hidden, penal, weights = self.txt_enc(captions, lengths, hidden)

        embed = BM.view(captions.size()[0], self.r, -1)
        noise = torch.FloatTensor(embed.size(0), embed.size(1), self.noise_dim).cuda()
        noise.data.normal_(0, 1)

        stage2_outputs, stage1_outputs, mus, logvars = self.upsample_block(embed, noise)
        images = self.image(stage2_outputs)

        return images, penal, hidden, mus
