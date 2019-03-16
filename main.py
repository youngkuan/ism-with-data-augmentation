# -*- coding: utf-8 -*-

####################### 主函数 ############################
from train import Trainer

if __name__ == '__main__':
    arguments = {}

    arguments['epochs'] = 8
    arguments['batch_size'] = 4
    arguments['num_workers'] = 0
    arguments['learning_rate'] = 0.001
    arguments['beta1'] = 0.5
    arguments['l1_coef'] = 1
    arguments['margin'] = 0.5

    arguments['image_path'] = "../data/flickr8k/images"
    arguments['sentence_path'] = "../data/flickr8k/Flickr8k.token.txt"

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



    trainer = Trainer(arguments)
    trainer.train()
