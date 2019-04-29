# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model import EncoderText
from upsample import UpsampleNetwork


class ImageGenerator(nn.Module):

    def __init__(self, arguments):
        super(ImageGenerator, self).__init__()
        self.noise_dim = arguments['noise_dim']
        self.embed_size = arguments['embed_size']

        self.txt_enc = EncoderText(arguments['vocab_size'], arguments['word_dim'],
                                   arguments['embed_size'], arguments['num_layers'],
                                   use_bi_gru=arguments['bi_gru'],
                                   no_txtnorm=arguments['no_txtnorm'])

        self.upsample_block = UpsampleNetwork(arguments)

    def generate_segment_emb(self, captions, segments):
        chunks = []
        for i, segment in enumerate(segments):
            segment_embs = []
            for j, s in enumerate(segment):
                begin = s[0] + 1
                end = s[1] + 1
                segment_embs.append(torch.mean(captions[i, begin:end, :], 1))

            segment_embs = torch.stack(segment_embs, 0)
            segment_embs = segment_embs.view(1, -1, self.embed_size)
            chunks.append(segment_embs)
            chunks = torch.stack(chunks, 0)
        return chunks


    def forward(self, captions, segments, lengths):

        # cap_emb -> batch_size*word_count*embed_size
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        print "cap_emb: ", cap_emb.size()

        word_emb = cap_emb[:, -1, :].view(cap_emb.size()[0],1,-1)
        print "word_emb: ", word_emb.size()
        segment_embs = self.generate_segment_emb(cap_emb, segments)
        print "segment_embs: ", segment_embs.size()
        sentence_emb = torch.mean(cap_emb,1).view(cap_emb.size()[0],1,-1)
        print "sentence_emb: ", sentence_emb.size()

        embed = torch.stack([word_emb,segment_embs,sentence_emb],1)
        noise = torch.FloatTensor(embed.size(0), self.noise_dim).cuda()
        noise.data.normal_(0, 1)
        images, mus, logvars = self.upsample_block(embed, noise)

        return images, cap_lens, mus, logvars


