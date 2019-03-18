# -*- coding: utf-8 -*-

####################### 性能评估 ############################
# 1. 图像检索文本召回率
# 1. 文本检索图像召回率

import numpy as np
import torch


def i2t(discriminator, images, sentences):
    """
    :param discriminator: 匹配判别器
    :param images: batch_size*3*224*224
    :param sentences: batch_size*sentence_max_length
    :return:
    """
    size = images.size()[0]
    ranks = np.zeros(size)
    for index in range(size):
        current_image = images[index].unsqueeze(0)
        current_images = current_image.expand(5 * size, current_image.size()[1], current_image.size()[2],
                                              current_image.size()[3])
        scores = discriminator(current_images, sentences)
        sort_s, indices = torch.sort(scores, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(indices == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
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
        current_sentences = current_sentence.expand(size / 5, current_sentence.size()[1])

        scores = discriminator(images, current_sentences)
        sort_s, indices = torch.sort(scores, descending=True)
        indices = indices.data.squeeze(0).cpu().numpy()
        ranks[index] = np.where(indices == (index / 5))[0][0]

        # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return r1, r5, r10, medr


if __name__ == '__main__':
    print ""
