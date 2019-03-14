# -*- coding: utf-8 -*-

####################### 数据加载器 ############################

import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import build_dictionary


def read_image(image_path, image_id):
    image_path = os.path.join(image_path, image_id)
    image = Image.open(image_path).resize((224, 224))
    return image


def read_sentences(sentence_path):
    sentences = []
    with open(sentence_path, 'rb') as f:
        for line in f:
            sentences.append(line.split("\t")[1])
    return sentences


class FakeImageDataLoader(Dataset):

    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['sentence_path']

        self.sentences = read_sentences(self.sentence_path)
        self.word_dictionary = build_dictionary(self.sentences)[0]
        self.word_number = len(self.word_dictionary)

        arguments['word_dictionary'] = self.word_dictionary
        arguments['word_number'] = self.word_number

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

        sentence = [self.word_dictionary[w] if self.word_dictionary[w] < self.word_number else 1 for w in
                    sentence.split()]

        sample = {
            'sentences': torch.Tensor(sentence),
            'matched_images': torch.FloatTensor(matched_image),
            'unmatched_images': torch.FloatTensor(unmatched_image)
        }
        sample['matched_images'] = sample['matched_images'].sub_(127.5).div_(127.5)
        sample['unmatched_images'] = sample['unmatched_images'].sub_(127.5).div_(127.5)
        return sample


class FakeSentenceDataLoader(Dataset):
    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['sentence_path']

        self.sentences = read_sentences(self.sentence_path)
        self.word_dictionary = build_dictionary(self.sentences)[0]
        self.word_number = len(self.word_dictionary)

        arguments['word_dictionary'] = self.word_dictionary
        arguments['word_number'] = self.word_number

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

        matched_sentence = [self.word_dictionary[w] if self.word_dictionary[w] < self.word_number else 1 for w in
                            matched_sentence.split()]
        unmatched_sentence = [self.word_dictionary[w] if self.word_dictionary[w] < self.word_number else 1 for w in
                              unmatched_sentence.split()]

        sample = {
            'images': torch.FloatTensor(image),
            'matched_sentences': torch.Tensor(matched_sentence),
            'unmatched_sentences': torch.Tensor(unmatched_sentence)
        }
        sample['images'] = sample['images'].sub_(127.5).div_(127.5)
        return sample


class MatchDataLoader(Dataset):
    def __init__(self, arguments):
        self.image_path = arguments['image_path']
        self.sentence_path = arguments['sentence_path']

        self.sentences = read_sentences(self.sentence_path)
        self.word_dictionary = build_dictionary(self.sentences)[0]
        self.word_number = len(self.word_dictionary)

        arguments['word_dictionary'] = self.word_dictionary
        arguments['word_number'] = self.word_number

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

        sentence = [self.word_dictionary[w] if self.word_dictionary[w] < self.word_number else 1 for w in
                    sentence.split()]
        unmatched_sentence = [self.word_dictionary[w] if self.word_dictionary[w] < self.word_number else 1 for w in
                              unmatched_sentence.split()]

        sample = {
            'images': torch.FloatTensor(image),
            'sentences': torch.Tensor(sentence),
            'unmatched_images': torch.Tensor(unmatched_image),
            'unmatched_sentences': torch.Tensor(unmatched_sentence)
        }
        sample['images'] = sample['images'].sub_(127.5).div_(127.5)
        sample['unmatched_images'] = sample['unmatched_images'].sub_(127.5).div_(127.5)
        return sample

    if __name__ == '__main__':
        sentences = ["Girl jumping rope in parking lot", "Girl jumping rope in parking lot",
                     "Girl jumping rope in parking lot"]
