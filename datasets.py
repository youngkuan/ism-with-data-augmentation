# -*- coding: utf-8 -*-

####################### 数据加载器 ############################

import os
import random
import numpy as np
import nltk
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import load_boxes, load_train_ids, load_image_name_to_id


class GeneratorDataset(Dataset):
    def __init__(self, arguments, vocab):
        self.data_dir = arguments['data_dir']
        self.caption_file = arguments['caption_file']
        self.train_caption_file = arguments['train_caption_file']
        self.val_caption_file = arguments['val_caption_file']
        self.train_id_file = arguments['train_id_file']
        self.image_feature_file = arguments['image_feature_file']
        self.box_file = arguments['box_file']
        self.image_size = arguments["image_size"]
        self.region_size = arguments["region_size"]
        self.image_path = os.path.join(self.data_dir, "images")
        self.train_path = os.path.join(self.data_dir, "train")
        self.im_div = 5
        self.vocab = vocab
        self.region_transform = transforms.Compose([
            transforms.Resize((self.region_size, self.region_size), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data
        print "----------------load image captions--------------"
        # Captions
        self.captions = []
        with open(os.path.join(self.train_path,self.caption_file), 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
        print "----------------load image boxes--------------"
        self.boxes = load_boxes(self.train_path, self.box_file)
        print "----------------load image ids--------------"
        self.image_ids = load_train_ids(self.train_path, self.train_id_file)
        print "----------------load image name to ids--------------"
        self.imageNameToId, self.imageIdToName = load_image_name_to_id(self.train_path, self.train_caption_file,
                                                                       self.val_caption_file)

        self.length = len(self.captions)
        print "dataset size: ", self.length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index / self.im_div

        caption = self.captions[index]
        vocab = self.vocab

        image_id = self.image_ids[img_index]
        image_name = self.imageIdToName[image_id]
        image_name = '%s/%s' % (self.image_path, image_name)
        image = Image.open(image_name).convert('RGB')

        box = self.boxes[img_index]
        regions = []
        for i, b in enumerate(box):
            region = image.crop((b[0], b[1], b[2], b[3]))
            region.save("images/%s_%d.jpg" % (img_index, i))
            region = self.region_transform(region)
            regions.append(region)
        regions = torch.stack(regions, dim=0)

        image = self.image_transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        return image, regions, caption, index, img_index


class DiscriminatorDataset(Dataset):
    def __init__(self, arguments, data_split, vocab):
        self.data_dir = arguments['data_dir']
        self.image_path = os.path.join(self.data_dir, "images")
        self.train_path = os.path.join(self.data_dir, "train")
        self.caption_file = arguments['caption_file']
        self.image_feature_file = arguments['image_feature_file']

        self.vocab = vocab

        # load data
        print "----------------load image captions--------------"
        # Captions
        self.captions = []
        with open(os.path.join(self.train_path, '%s_caps.txt' % data_split), 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        print "----------------load image features--------------"
        self.images = np.load(os.path.join(self.train_path, '%s_ims.npy' % data_split))

        self.length = len(self.captions)

        self.im_div = 5

        if data_split == 'dev':
            self.length = 5000


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index / self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, img_id


def collate_GData(data):
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
    images, regions, captions, indexes, img_indexes = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    regions = torch.stack(regions, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, regions, targets, lengths, indexes

def collate_DData(data):
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
    images, captions, indexes, img_indexes = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, indexes




def get_gan_loaders(arguments, vocab, batch_size, num_workers):
    dataset = GeneratorDataset(arguments, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, collate_fn=collate_GData)

    return loader

def get_loader(arguments, vocab, split_name):

    batch_size = arguments['batch_size']
    num_workers = arguments['num_workers']
    dataset = DiscriminatorDataset(arguments, split_name, vocab)
    loader = DataLoader(dataset, batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers, collate_fn=collate_DData)
    return loader
