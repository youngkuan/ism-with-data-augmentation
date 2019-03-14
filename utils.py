# -*- coding: utf-8 -*-


import numpy
from collections import OrderedDict

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    word_count = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in word_count:
                word_count[w] = 0
                word_count[w] += 1
    words = word_count.keys()
    freqs = word_count.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    word_dictionary = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        word_dictionary[words[sidx]] = idx + 2   # 0: <eos>, 1: <unk>

    word_dictionary['<eos>'] = 0
    word_dictionary['UNK'] = 1

    return word_dictionary, word_count


if __name__ == '__main__':
    sentences = ["Girl jumping rope in parking lot", "Girl jumping rope in parking lot",
                 "Girl jumping rope in parking lot"]
    word_dictionary, word_count = build_dictionary(sentences)
    n_words = 10000

    seqs = []
    for i, cc in enumerate(sentences):
        seqs.append([word_dictionary[w] if word_dictionary[w] < n_words else 1 for w in cc.split()])

    print seqs
    lengths = [len(s) for s in seqs]
    print "lengths: ",lengths
    n_samples = len(seqs)
    maxlen = numpy.max(lengths) + 1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        print "s: ",s
        x[:lengths[idx], idx] = s

    print x