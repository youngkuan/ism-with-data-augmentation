# -*- coding: utf-8 -*-

####################### 对特征进行下decoder，生成文本 ############################
# 使用GRU进行decoder
# reference
# https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceDecoder(nn.Module):
    def __init__(self, arguments):
        super(SentenceDecoder, self).__init__()
        self.hidden_size = arguments['hidden_size']
        self.output_size = arguments['output_size']

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
