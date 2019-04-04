# -*- coding: utf-8 -*-

####################### 主函数 ############################

from train import Trainer
from utils import build_dictionary

if __name__ == '__main__':
    arguments = {}

    arguments['epochs'] = 5
    arguments['batch_size'] = 32
    arguments['num_workers'] = 0
    arguments['learning_rate'] = 0.02
    arguments['beta1'] = 0.5
    arguments['kl_coef'] = 2
    arguments['margin'] = 0.5
    arguments['lr_decay_step'] = 20
    arguments["image_size"] = 224

    arguments['data_dir'] = "../data/mscoco2014"
    arguments['synthetic_image_path'] = "../data/mscoco2014/synthetic_images"
    arguments['model_save_path'] = "./models"
    arguments['loss_save_path'] = "./loss"


    # 上采样参数（upsample）
    arguments['ngf'] = 192 * 8
    arguments['num_channels'] = 3
    arguments['image_feature_size'] = 512
    arguments["sentence_embedding_size"] = 1024
    arguments["condition_dimension"] = 128
    # 噪声维度
    arguments['noise_dim'] = 100

    trainer = Trainer(arguments)
    trainer.train()
