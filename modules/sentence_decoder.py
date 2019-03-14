# -*- coding: utf-8 -*-

####################### 对特征进行下decoder，生成文本 ############################
# 使用GRU进行decoder
# reference
# https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# https://github.com/nikhilmaram/Show_and_Tell
# https://github.com/yunjey/pytorch-tutorial
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentenceDecoder(nn.Module):
    def __init__(self, arguments):
        super(SentenceDecoder, self).__init__()
        self.embed_size = arguments['embed_size']
        self.hidden_size = arguments['hidden_size']
        self.word_number = arguments['word_number']
        self.num_layers = arguments['num_layers']

        self.embedding = nn.Embedding(self.word_number, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.word_number)

        self.max_seg_length = arguments['max_seq_length']

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
