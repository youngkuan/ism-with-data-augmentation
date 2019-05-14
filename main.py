# -*- coding: utf-8 -*-

####################### 主函数 ############################

from evaluation import evaluate
from train import Trainer

if __name__ == '__main__':
    arguments = {}

    arguments['gpus'] = [0, 1]

    arguments['epochs'] = 120
    arguments['batch_size'] = 16
    arguments['num_workers'] = 2
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

    # file
    arguments['voc_file'] = "coco_precomp_vocab.json"
    arguments['caption_file'] = "train_caps.txt"
    arguments['train_caption_file'] = "captions_train2014.json"
    arguments['val_caption_file'] = "captions_val2014.json"
    arguments['image_feature_file'] = "train_ims.npy"
    arguments['train_id_file'] = "train_ids.txt"
    arguments['box_file'] = "train_boxes.npy"


    # 上采样参数（upsample）
    arguments['ngf'] = 192 * 8
    arguments['ndf'] = 96
    arguments['num_channels'] = 3
    arguments['image_feature_size'] = 512
    arguments["embed_size"] = 1024
    arguments["word_dim"] = 300
    arguments['num_layers'] = 1
    arguments['bi_gru'] = True
    arguments['no_txtnorm'] = True
    arguments["project_size"] = 512
    arguments["condition_dimension"] = 128
    # 判别器是否加入条件信息
    arguments["bcondition"]  = False
    # 噪声维度
    arguments['noise_dim'] = 100

    trainer = Trainer(arguments)
    trainer.train()

    evaluate(arguments)
