# -*- coding: utf-8 -*-

####################### 数据加载器 ############################

import os
import random

from torch.utils.data import Dataset

from utils import get_image, load_filenames, load_embedding, load_class_id


class FakeImageDataLoader(Dataset):
    def __init__(self, arguments, transform=None, embedding_type='cnn-rnn'):
        self.data_dir = arguments['data_dir']
        self.image_path = os.path.join(self.data_dir, "images")
        self.train_path = os.path.join(self.data_dir, "train")
        self.embedding_type = embedding_type

        self.filenames = load_filenames(self.train_path)
        self.embeddings = load_embedding(self.train_path, embedding_type)
        self.class_id = load_class_id(self.train_path, len(self.filenames))

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        embeddings = self.embeddings[index, :, :]
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embeddings = embeddings[embedding_ix, :]

        image_name = '%s/%s.jpg' % (self.image_path, key)
        images = get_image(image_name, self.transform)

        unmatched_index = random.randint(0, len(self.filenames) - 1)
        while abs(unmatched_index - index) < 10:
            unmatched_index = random.randint(0, len(self.filenames) - 1)

        unmatched_image_name = '%s/%s.jpg' % (self.image_path, self.filenames[unmatched_index])
        unmatched_images = get_image(unmatched_image_name, self.transform)

        sample = {
            'embeddings': embeddings,
            'images': images,
            'unmatched_images': unmatched_images
        }
        return sample


class MatchDataLoader(Dataset):
    def __init__(self, arguments, transform=None, embedding_type='cnn-rnn'):
        self.data_dir = arguments['data_dir']
        self.image_path = os.path.join(self.data_dir, "images")
        self.train_path = os.path.join(self.data_dir, "train")
        self.embedding_type = embedding_type

        self.filenames = load_filenames(self.train_path)
        self.embeddings = load_embedding(self.train_path, embedding_type)
        self.class_id = load_class_id(self.train_path, len(self.filenames))

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        key = self.filenames[index]
        embeddings = self.embeddings[index, :, :]
        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embeddings = embeddings[embedding_ix, :]

        image_name = '%s/%s.jpg' % (self.image_path, key)
        images = get_image(image_name, self.transform)

        unmatched_index = random.randint(0, len(self.filenames) - 1)
        while abs(unmatched_index - index) < 10:
            unmatched_index = random.randint(0, len(self.filenames) - 1)

        unmatched_image_name = '%s/%s.jpg' % (self.image_path, self.filenames[unmatched_index])
        unmatched_images = get_image(unmatched_image_name, self.transform)

        sample = {
            'embeddings': embeddings,
            'images': images,
            'unmatched_images': unmatched_images
        }
        return sample
