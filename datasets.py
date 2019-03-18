# -*- coding: utf-8 -*-

####################### 数据加载器 ############################

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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


class FakeImageDataLoader(Dataset):
    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['train_sentence_path']

        self.word2idx = arguments['word2idx']
        self.idx2word = arguments['idx2word']
        self.lengths = arguments['lengths']
        self.word_number = arguments['word_number']
        self.sentence_max_length = arguments['sentence_max_length']

    def __len__(self):
        with open(self.sentence_path, 'rb') as f:
            length = len(f.readlines())
        return length

    def __getitem__(self, index):
        with open(self.sentence_path, 'rb') as f:
            lines = f.readlines()
            line = lines[index]
            sentence = line.split("\t")[1]
            matched_image_id = line.split("#")[0]

            unmatched_index = random.randint(0, len(lines))
            while abs(unmatched_index - index) < 10:
                unmatched_index = random.randint(0, len(lines))

            unmatched_image_id = lines[unmatched_index].split("#")[0]

        matched_image = read_image(self.image_path, matched_image_id)
        unmatched_image = read_image(self.image_path, unmatched_image_id)

        sentence = [self.word2idx[w] if self.word2idx[w] < self.word_number else 1 for w in
                    sentence.split()]
        x = np.zeros(self.sentence_max_length).astype('int64')
        x[:len(sentence)] = sentence

        sample = {
            'sentences': torch.LongTensor(x),
            'matched_images': torch.FloatTensor(matched_image),
            'unmatched_images': torch.FloatTensor(unmatched_image)
        }
        sample['matched_images'] = sample['matched_images'].sub_(127.5).div_(127.5)
        sample['unmatched_images'] = sample['unmatched_images'].sub_(127.5).div_(127.5)
        return sample


class FakeSentenceDataLoader(Dataset):
    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['train_sentence_path']

        self.word2idx = arguments['word2idx']
        self.idx2word = arguments['idx2word']
        self.lengths = arguments['lengths']
        self.word_number = arguments['word_number']
        self.sentence_max_length = arguments['sentence_max_length']

    def __len__(self):
        with open(self.sentence_path, 'rb') as f:
            length = len(f.readlines())
        return length

    def __getitem__(self, index):
        with open(self.sentence_path, 'rb') as f:
            lines = f.readlines()
            line = lines[index]

            image_id = line.split("#")[0]
            matched_sentence = line.split("\t")[1]

            unmatched_index = random.randint(0, len(lines))
            while abs(unmatched_index - index) < 10:
                unmatched_index = random.randint(0, len(lines))

            unmatched_sentence = lines[unmatched_index].split("\t")[1]

        image = read_image(self.image_path, image_id)

        matched_sentence = [self.word2idx[w] if self.word2idx[w] < self.word_number else 1 for w in
                            matched_sentence.split()]
        matched_x = np.zeros(self.sentence_max_length).astype('int64')
        matched_x[:len(matched_sentence)] = matched_sentence

        unmatched_sentence = [self.word2idx[w] if self.word2idx[w] < self.word_number else 1 for w in
                              unmatched_sentence.split()]
        unmatched_x = np.zeros(self.sentence_max_length).astype('int64')
        unmatched_x[:len(unmatched_sentence)] = unmatched_sentence

        sample = {
            'images': torch.FloatTensor(image),
            'matched_sentences': torch.LongTensor(matched_x),
            'unmatched_sentences': torch.LongTensor(unmatched_x)
        }
        sample['images'] = sample['images'].sub_(127.5).div_(127.5)
        return sample


class MatchDataLoader(Dataset):
    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['train_sentence_path']

        self.word2idx = arguments['word2idx']
        self.idx2word = arguments['idx2word']
        self.lengths = arguments['lengths']
        self.word_number = arguments['word_number']
        self.sentence_max_length = arguments['sentence_max_length']

    def __len__(self):
        with open(self.sentence_path, 'rb') as f:
            length = len(f.readlines())
        return length

    def __getitem__(self, index):
        with open(self.sentence_path, 'rb') as f:
            lines = f.readlines()
            line = lines[index]

            image_id = line.split("#")[0]
            sentence = line.split("\t")[1]

            unmatched_index = random.randint(0, len(lines))
            while abs(unmatched_index - index) < 10:
                unmatched_index = random.randint(0, len(lines))

            unmatched_image_id = lines[unmatched_index].split("#")[0]
            unmatched_sentence = lines[unmatched_index].split("\t")[1]

        image = read_image(self.image_path, image_id)
        unmatched_image = read_image(self.image_path, unmatched_image_id)

        sentence = [self.word2idx[w] if self.word2idx[w] < self.word_number else 1 for w in
                    sentence.split()]

        x = np.zeros(self.sentence_max_length).astype('int64')
        x[:len(sentence)] = sentence

        unmatched_sentence = [self.word2idx[w] if self.word2idx[w] < self.word_number else 1 for w in
                              unmatched_sentence.split()]
        unmatched_x = np.zeros(self.sentence_max_length).astype('int64')
        unmatched_x[:len(unmatched_sentence)] = unmatched_sentence

        sample = {
            'images': torch.FloatTensor(image),
            'sentences': torch.LongTensor(x),
            'unmatched_images': torch.FloatTensor(unmatched_image),
            'unmatched_sentences': torch.LongTensor(unmatched_x)
        }
        sample['images'] = sample['images'].sub_(127.5).div_(127.5)
        sample['unmatched_images'] = sample['unmatched_images'].sub_(127.5).div_(127.5)
        return sample
