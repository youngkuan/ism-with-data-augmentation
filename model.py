# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        total_length = x.size(1)
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True, total_length=total_length)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class SelfAttentive(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, da, r, emb_matrix, cuda,
                 use_bi_gru=True):
        super(SelfAttentive, self).__init__()

        # Embedding Layer
        self.encoder = nn.Embedding(vocab_size, word_dim)

        # RNN type
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        # Self Attention Layers
        self.S1 = nn.Linear(embed_size * 2, da, bias=False)
        self.S2 = nn.Linear(da, r, bias=False)


        self.init_wordembedding(emb_matrix)
        self.init_weights()

        self.r = r
        self.nhid = embed_size
        self.nlayers = embed_size

        if cuda:
            self.cuda()

    def init_weights(self):
        initrange = 0.1
        self.S1.weight.data.uniform_(-initrange, initrange)
        self.S2.weight.data.uniform_(-initrange, initrange)

    def init_wordembedding(self, embedding_matrix):
        self.encoder.weight.data = embedding_matrix

    def forward(self, input, hidden, lengths):

        emb = self.encoder(input)

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        output, hidden = self.rnn(packed, hidden)

        depacked_output, lens = pad_packed_sequence(output, batch_first=True, total_length=input.size()[0])

        if self.cuda:
            BM = torch.zeros(input.size(0), self.r * self.nhid * 2).cuda()
            penal = torch.zeros(1).cuda()
            I = torch.eye(self.r).cuda()
        else:
            BM = torch.zeros(input.size(0), self.r * self.nhid * 2)
            penal = torch.zeros(1)
            I = torch.eye(self.r)
        weights = {}

        # Attention Block
        for i in range(input.size(0)):
            H = depacked_output[i, :lens[i], :]
            s1 = self.S1(H)
            s2 = self.S2(functional.tanh(s1))

            # Attention Weights and Embedding
            A = functional.softmax(s2.t())
            M = torch.mm(A, H)
            BM[i, :] = M.view(-1)

            # Penalization term
            AAT = torch.mm(A, A.t())
            P = torch.norm(AAT - I, 2)
            penal += P * P
            weights[i] = A

        # Penalization Term
        penal /= input.size(0)

        return BM, hidden, penal, weights

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (weight.new(self.nlayers * 2, bsz, self.nhid).zero_(),
                weight.new(self.nlayers * 2, bsz, self.nhid).zero_())
