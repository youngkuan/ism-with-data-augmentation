# -*- coding: utf-8 -*-


import os
import pickle
from collections import OrderedDict
import numpy
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from modules.discrminator import MatchDiscriminator


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def save_discriminator_checkpoint(discriminator, model_save_path, epoch):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(discriminator.state_dict(), '{0}/disc_{1}.pth'.format(model_save_path, epoch))


def load_discriminator(discriminator_model_path, arguments):
    # load discriminator model
    discriminator = MatchDiscriminator(arguments).cuda()
    discriminator.load_state_dict(torch.load(discriminator_model_path))
    return discriminator


def split_train_validation_set(sentence_path, train_path, val_path, val_number=500):
    """
    拆分训练集和验证集，并保存到文件
    :param sentence_path:
    :param train_path
    :param val_path:
    :param val_number:
    :return:
    """
    with open(sentence_path, 'rb') as f:
        lines = f.readlines()
        validation_set = lines[:val_number]
        train_set = lines[val_number:]
    with open(train_path, 'w') as train_file:
        train_file.writelines(train_set)
    with open(val_path, 'w') as val_file:
        val_file.writelines(validation_set)


def read_image(image_path, image_id):
    image_path = os.path.join(image_path, image_id)
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image)
    return np.transpose(image, (2, 0, 1))


def read_sentences(sentence_path):
    sentences = []
    with open(sentence_path, 'rb') as f:
        for line in f:
            sentences.append(line.split("\t")[1])
    return sentences


def load_validation_set(arguments):
    word2idx = arguments["word2idx"]
    word_number = arguments['word_number']
    sentence_max_length = arguments['sentence_max_length']
    image_path = arguments['image_path']
    sentence_path = arguments['val_sentence_path']
    sentences = []
    images = []
    last_image_id = ""
    with open(sentence_path, 'rb') as f:
        for line in f:
            sentence = line.split("\t")[1]
            sentence = [word2idx[w] if word2idx[w] < word_number else 1 for w in sentence.split()]
            x = np.zeros(sentence_max_length).astype('int64')
            x[:len(sentence)] = sentence
            sentences.append(x)

            image_id = line.split("#")[0]
            if last_image_id != image_id:
                image = read_image(image_path, image_id)
                images.append(image)
                last_image_id = image_id
    # n * 3*224*224
    images = np.stack(images, 0)
    # 5n * sentence_max_length
    sentences = np.stack(sentences, 0)

    return torch.FloatTensor(images), torch.LongTensor(sentences)


def load_embedding(data_dir, embedding_type):
    if embedding_type == 'cnn-rnn':
        embedding_filename = '/char-CNN-RNN-embeddings.pickle'
    elif embedding_type == 'cnn-gru':
        embedding_filename = '/char-CNN-GRU-embeddings.pickle'
    elif embedding_type == 'skip-thought':
        embedding_filename = '/skip-thought-embeddings.pickle'

    with open(data_dir + embedding_filename, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = np.array(embeddings)
        # embedding_shape = [embeddings.shape[-1]]
        print('embeddings: ', embeddings.shape)
    return embeddings


def load_filenames(data_dir):
    filepath = os.path.join(data_dir, 'filenames.pickle')
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_class_id(data_dir, total_num):
    if os.path.isfile(data_dir + '/class_info.pickle'):
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f)
    else:
        class_id = np.arange(total_num)
    return class_id


def get_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image


def collate_fn(data):
    """Create mini-batches of (image, caption)

    Custom collate_fn for torch.utils.data.DataLoader is necessary for patting captions

    :param data: list; (image, caption) tuples
            - image: tensor;    3 x 256 x 256
            - caption: tensor;  1 x length_caption

    Return: mini-batch
    :return images: tensor;     batch_size x 3 x 256 x 256
    :return padded_captions: tensor;    batch_size x length
    :return caption_lengths: list;      lenghths of actual captions (without padding)
    """

    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge image tensors (stack)
    images = torch.stack(images, 0)

    # Merge captions
    caption_lengths = [len(caption) for caption in captions]

    # zero-matrix num_captions x caption_max_length
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()

    # fill the zero-matrix with captions. the remaining zeros are padding
    for ix, caption in enumerate(captions):
        end = caption_lengths[ix]
        padded_captions[ix, :end] = caption[:end]
    return images, padded_captions, caption_lengths


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


def save_image(image, synthetic_image_path, image_name):
    im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
    im.save('{0}/{1}'.format(synthetic_image_path, image_name))


def save_img_results(data_img, fake, epoch, image_dir, batch_size):
    num = batch_size
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
                       (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
                       (image_dir, epoch), normalize=True)


def save_sentence(val_sentences, fake_sentences, synthetic_sentence_path):
    with open(synthetic_sentence_path, "w") as f:
        f.writelines(val_sentences)
        f.writelines(fake_sentences)


def save_loss(save_loss, loss_save_path):
    """
    save result
    :param results:
    :param result_save_path:
    :return:
    """
    txt_path = os.path.join(loss_save_path, 'loss.txt')

    if os.path.exists(txt_path):
        os.remove(txt_path)
    np.savetxt(txt_path, save_loss)


def convert_indexes2sentence(idx2word, sentences_indexes):
    sentences = []
    for sentence_indexes in sentences_indexes:
        words = []
        for word_id in sentence_indexes:
            word = idx2word[word_id.item()]
            words.append(word)
            if word == '<eos>':
                break
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
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
    maxlen = numpy.max(lengths) + 1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s

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
