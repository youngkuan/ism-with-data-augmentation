# -*- coding: utf-8 -*-

####################### 主函数 ############################

from train import Trainer
from evaluation import evaluate
from PIL import Image


if __name__ == '__main__':
    arguments = {}

    arguments['epochs'] = 120
    arguments['batch_size'] = 128
    arguments['num_workers'] = 0
    arguments['learning_rate'] = 0.0002
    arguments['beta1'] = 0.5
    arguments['kl_coef'] = 2
    arguments['margin'] = 0.5
    arguments['lr_decay_step'] = 20
    arguments["image_size"] = 64

    arguments['data_dir'] = "../data/mscoco2014"
    arguments['val_image_path'] = "../data/mscoco2014/val_images"
    arguments['train_path'] = "../data/mscoco2014/train"
    arguments['val_path'] = "../data/mscoco2014/val"
    arguments['synthetic_image_path'] = "../data/mscoco2014/synthetic_images"
    arguments['model_save_path'] = "./models"
    arguments['loss_save_path'] = "./loss"


    # 上采样参数（upsample）
    arguments['ngf'] = 192 * 8
    arguments['ndf'] = 96
    arguments['num_channels'] = 3
    arguments['image_feature_size'] = 512
    arguments["sentence_embedding_size"] = 1024
    arguments["project_size"] = 512
    arguments["condition_dimension"] = 128
    # 噪声维度
    arguments['noise_dim'] = 100

    trainer = Trainer(arguments)
    trainer.train()

    evaluate(arguments)
