# -*- coding: utf-8 -*-

####################### 数据加载器 ############################

import os
import random

import nltk
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from modules.utils import get_image, load_filenames, load_embedding, load_class_id
from modules.utils import load_image_name_to_id, load_train_ids, \
    segment_sentence_to_chunk


class GeneratorDataset(Dataset):
    def __init__(self, arguments, vocab, image_transform):
        self.data_dir = arguments['data_dir']
        self.caption_file = arguments['caption_file']
        self.train_caption_file = arguments['train_caption_file']
        self.val_caption_file = arguments['val_caption_file']
        self.train_id_file = arguments['train_id_file']
        self.image_feature_file = arguments['image_feature_file']
        self.image_path = os.path.join(self.data_dir, "images")
        self.train_path = os.path.join(self.data_dir, "train")
        self.im_div = 5
        self.vocab = vocab
        self.transform = image_transform

        # load data
        print "----------------load image features--------------"
        # self.image_features = load_image_features(self.train_path, self.image_feature_file)
        self.image_features = None
        print "----------------load image captions and segments--------------"
        self.segments,self.captions = segment_sentence_to_chunk(self.train_path, self.caption_file)
        print "----------------load image ids--------------"
        self.image_ids = load_train_ids(self.train_path, self.train_id_file)
        print "----------------load image ids 2 image names--------------"
        self.imageNameToId, self.imageIdToName = load_image_name_to_id(self.train_path, self.train_caption_file,
                                                                       self.val_caption_file)

        self.length = len(self.captions)
        print "dataset size: ", self.length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index / self.im_div
        # image_feature = torch.Tensor(self.images[img_index])
        image_feature = None
        caption = self.captions[index]
        segment = self.segments[index]
        vocab = self.vocab

        image_id = self.image_ids[img_index]
        image_name = self.imageIdToName[image_id]
        image_name = '%s/%s' % (self.image_path, image_name)
        image = get_image(image_name, self.transform)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)
        return image, image_feature, caption, segment, index, img_index


class DiscriminatorDataset(Dataset):
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


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    images, image_features, captions, segments, indexes, img_indexes = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # image_features = torch.stack(image_features, 0)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, image_features, targets, segments, lengths, indexes


def get_loaders(arguments, vocab, batch_size, num_workers, image_transform=None):
    generator_dataset = GeneratorDataset(arguments, vocab, image_transform)
    generator_data_loader = DataLoader(generator_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=collate_fn)

    match_dataset = DiscriminatorDataset(arguments)
    match_data_loader = DataLoader(match_dataset, batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)

    return generator_data_loader, match_data_loader
