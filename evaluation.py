# -*- coding: utf-8 -*-

####################### 性能评估 ############################
# 1. 图像检索文本召回率
# 1. 文本检索图像召回率

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils import load_discriminator, load_val_embedding, load_val_filenames


def i2t(discriminator, images, sentences):
    """
    :param discriminator: 匹配判别器
    :param images: N*3*224*224
    :param sentences: N * embedding_size
    :return:
    """
    size = images.size()[0]
    ranks = np.zeros(size)
    for index in range(size):
        current_image = images[index].unsqueeze(0)
        current_images = current_image.expand(size, current_image.size()[1], current_image.size()[2],
                                              current_image.size()[3])
        scores = discriminator(current_images, sentences)
        sort_s, indices = torch.sort(scores, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        # Score
        # find the highest ranking
        ranks[index] = np.where(indices == index)[0][0]
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return r1, r5, r10, medr


def t2i(discriminator, sentences, images):
    """
    :param discriminator:匹配判别器
    :param sentences:
    :param images:
    :return:
    """
    size = sentences.size()[0]
    ranks = np.zeros(size)
    for index in range(size):
        current_sentence = sentences[index].unsqueeze(0)
        current_sentences = current_sentence.expand(size, current_sentence.size()[1])

        scores = discriminator(images, current_sentences)
        sort_s, indices = torch.sort(scores, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        ranks[index] = np.where(indices == index)[0][0]

        # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return r1, r5, r10, medr


def evaluate(arguments):
    model_save_path = arguments["model_save_path"]
    epoch = arguments['epochs'] - 1
    image_size = arguments['image_size']
    image_path = arguments['val_image_path']
    val_path = arguments['val_path']
    embedding_type = 'cnn-rnn'
    val_number = 100

    match_discriminator = load_discriminator('{0}/disc_{1}.pth'.format(model_save_path, epoch), arguments)

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    filenames = load_val_filenames(val_path)
    embeddings = load_val_embedding(val_path)
    embeddings = embeddings[:val_number]
    embeddings = torch.tensor(embeddings)

    images = []
    for i in range(val_number):
        image_name = '%s/%s.jpg' % (image_path, filenames[i])
        image = Image.open(image_name).convert('RGB')
        image = image_transform(image)
        images.append(image)

    print "load images done "
    images = torch.stack(images, dim=0)
    i2t_r1, i2t_r5, i2t_r10, i2t_medr = i2t(match_discriminator, images, embeddings)
    t2i_r1, t2i_r5, t2i_r10, t2i_medr = t2i(match_discriminator, embeddings, images)
    print "Image to Text: %.2f, %.2f, %.2f, %.2f" \
          % (i2t_r1, i2t_r5, i2t_r10, i2t_medr)
    print "Text to Image: %.2f, %.2f, %.2f, %.2f" \
          % (t2i_r1, t2i_r5, t2i_r10, t2i_medr)


if __name__ == '__main__':
    print ""
