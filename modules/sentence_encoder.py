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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceEncoder(nn.Module):

    def __init__(self, arguments):
        # init all parameters
        super(SentenceEncoder, self).__init__()
        self.hidden_size = arguments['hidden_size']
        self.word_dimension = arguments['word_dimension']
        self.batch_size = arguments['batch_size']

        self.embedding = nn.Embedding(arguments['word_number'], self.word_dimension)
        self.gru = nn.GRU(self.word_dimension, self.hidden_size,batch_first=True)
        self.sentence_max_length = arguments['sentence_max_length']

    def forward(self, input, encoder_hidden):
        embedded = self.embedding(input)

        output, hidden = self.gru(embedded, encoder_hidden)
        hidden = torch.squeeze(hidden)
        return hidden

    def initHidden(self):
        return torch.zeros(1 , self.batch_size, self.hidden_size, device=device)
