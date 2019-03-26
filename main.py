# -*- coding: utf-8 -*-

####################### 主函数 ############################

import numpy as np
from datasets import read_sentences,split_train_validation_set
from train import Trainer
from utils import build_dictionary


if __name__ == '__main__':
    arguments = {}

    arguments['epochs'] = 32
    arguments['batch_size'] = 128
    arguments['num_workers'] = 0
    arguments['learning_rate'] = 0.001
    arguments['beta1'] = 0.5
    arguments['l1_coef'] = 1
    arguments['margin'] = 0.5

    arguments['image_path'] = "../data/flickr8k/images"
    arguments['sentence_path'] = "../data/flickr8k/Flickr8k.token.txt"
    arguments['train_sentence_path'] = "../data/flickr8k/Flickr8k.token_train.txt"
    arguments['val_sentence_path'] = "../data/flickr8k/Flickr8k.token_val.txt"
    arguments['synthetic_image_path'] = "../data/flickr8k/synthetic_image"
    arguments['synthetic_sentence_path'] = "../data/flickr8k/synthetic_sentence.txt"

    arguments['model_save_path'] = "./models"

    # 文本编码参数（sentence encoder）
    # 词向量维度
    arguments['word_dimension'] = 512
    # 文本编码维度
    arguments['hidden_size'] = 512

    # 上采样参数（upsample）
    arguments['ngf'] = 64
    # 噪声维度
    arguments['noise_dim'] = 500
    # 上采样输入就是文本编码维度+噪声维度
    arguments['up_sample_input_dim'] = arguments['hidden_size'] + arguments['noise_dim']
    arguments['num_channels'] = 3

    # 下采样参数
    arguments['image_feature_size'] = 512

    # 文本解码参数
    arguments['hidden_size'] = 512
    arguments['num_layers'] = 1
    arguments["sentence_embedding_size"] = arguments['hidden_size']

    # 划分训练集和验证集
    split_train_validation_set(arguments['sentence_path'], arguments['train_sentence_path'],
                               arguments['val_sentence_path'], 200)
    # 建立词典
    sentences = read_sentences(arguments['sentence_path'])
    word2idx, idx2word, lengths = build_dictionary(sentences)
    arguments['word2idx'] = word2idx
    arguments['idx2word'] = idx2word
    arguments["lengths"] = lengths
    arguments['word_number'] = len(word2idx)
    arguments['max_seq_length'] = len(word2idx)
    arguments['sentence_max_length'] = np.max(lengths) + 1

    arguments['use_sentence_generator'] = False

    trainer = Trainer(arguments)
    trainer.train()
