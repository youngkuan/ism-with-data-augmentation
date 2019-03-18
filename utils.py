# -*- coding: utf-8 -*-


from collections import OrderedDict
from PIL import Image
import numpy
import torch.nn as nn


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

def save_image(image, synthetic_image_path,image_name):
    im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
    im.save('{0}/{1}'.format(synthetic_image_path, image_name))

def save_sentence(val_sentences, fake_sentences, synthetic_sentence_path):
    with open(synthetic_sentence_path,"w") as f:
        f.writelines(val_sentences)
        f.writelines(fake_sentences)

def convert_indexes2sentence(idx2word, sentences_indexes):
    sentences = []
    for sentence_indexes in sentences_indexes:
        words = []
        for word_id in sentence_indexes:
            word = idx2word[word_id]
            words.append(word)
            if word == '<eos>':
                break
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)



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
    # sentences = ["Girl jumping rope in parking lot", "Girl jumping rope in parking lot"
    #     , "Girl jumping rope parking", "A child going down an inflatable slide"]
    # word2idx, idx2word = build_dictionary(sentences)
    n_words = 10000
    # print word2idx
    # print idx2word
    # x,y = prepare_data(sentences,word2idx)
    # print x

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


