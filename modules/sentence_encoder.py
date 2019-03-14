# -*- coding: utf-8 -*-


############################### Sentence Encoder ####################################
# 使用GRU进行encoder
##### reference
# https://github.com/keishinkickback/Pytorch-RNN-text-classification
# skip-thoughts https://github.com/Cadene/skip-thoughts.torch/tree/master/pytorch
# https://github.com/linxd5/VSE_Pytorch
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import torch
import torch.nn as nn


class SentenceEncoder(nn.Module):

    def __init__(self, arguments):
        # init all parameters
        super(SentenceEncoder, self).__init__()
        self.hidden_size = arguments['hidden_size']
        self.input_size = arguments['word_number']

        self.embedding = nn.Embedding(arguments['word_number'], self.input_size)
        self.gru = nn.GRU(self.input_size, self.hidden_size)

    def forward(self, input):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, None)
        return output

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
