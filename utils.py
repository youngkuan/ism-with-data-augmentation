# -*- coding: utf-8 -*-


from collections import OrderedDict

import numpy


def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    word_count = OrderedDict()
    lengths = []
    for cc in text:
        words = cc.split()
        lengths.append(len(words))
        for w in words:
            if w not in word_count:
                word_count[w] = 0
            word_count[w] += 1
    words = word_count.keys()
    freqs = word_count.values()
    # 降序排序
    sorted_idx = numpy.argsort(freqs)[::-1]

    word2idx = OrderedDict()
    idx2word = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        word2idx[words[sidx]] = idx + 2  # 0: <eos>, 1: <unk>
        idx2word[idx + 2] = words[sidx]

    word2idx['<eos>'] = 0
    idx2word[0] = '<eos>'
    word2idx['UNK'] = 1
    idx2word[1] = 'UNK'

    return word2idx, idx2word, lengths


def prepare_data(caps, worddict, maxlen=None, n_words=10000):
    """
    Put data into format useable by the model
    """
    seqs = []
    feat_list = []
    for i, cc in enumerate(caps):
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])

    lengths = [len(s) for s in seqs]

    y = numpy.asarray(feat_list, dtype=numpy.float32)

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s

    return x, y


if __name__ == '__main__':
    sentences = ["Girl jumping rope in parking lot", "Girl jumping rope in parking lot"
        , "Girl jumping rope parking", "A child going down an inflatable slide"]
    word2idx, idx2word = build_dictionary(sentences)
    n_words = 10000
    print word2idx
    print idx2word
    x,y = prepare_data(sentences,word2idx)
    print x

    # seqs = []
    # for i, cc in enumerate(sentences):
    #     seq = torch.Tensor([word_dictionary[w] if word_dictionary[w] < n_words else 1 for w in cc.split()])
    #     seqs.append(seq)
    #
    # print seqs
    # lengths = [len(s) for s in seqs]
    # print "lengths: ",lengths
    # n_samples = len(seqs)
    # maxlen = numpy.max(lengths) + 1

    # x = numpy.zeros((maxlen, n_samples)).astype('int64')
    # for idx, s in enumerate(seqs):
    #     print "s: ",s
    #     x[:lengths[idx], idx] = s
    # seqs = torch.stack(seqs)
    # targets = pack_padded_sequence(seqs, lengths, batch_first=True)
    #
    # print targets


