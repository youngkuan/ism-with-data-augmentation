# -*- coding: utf-8 -*-


############################### Sentence Encoder ####################################
# 使用GRU进行encoder
##### reference
# https://github.com/keishinkickback/Pytorch-RNN-text-classification
# skip-thoughts https://github.com/Cadene/skip-thoughts.torch/tree/master/pytorch
# https://github.com/linxd5/VSE_Pytorch

import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):

    def __init__(self, arguments):
        # init all parameters
        super(SentenceEncoder, self).__init__()
        self.hidden_size = arguments['hidden_size']
        self.input_size = arguments['input_size']

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
