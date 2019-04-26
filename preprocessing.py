# -*- coding: utf-8 -*-

####################### 数据预处理 ############################
from modules.utils import reorder_image_feature_and_embedding
import os

def main():
    data_dir = "../data/mscoco2014"
    annotation_path = os.path.join(data_dir,"train")
    image_feature_file = "train_ims.npy"
    embedding_type = "cnn-rnn"
    caption_file = "captions.json"
    train_id_file = "train_ids.txt"
    reorder_image_feature_and_embedding(annotation_path, image_feature_file, embedding_type, caption_file,
                                        train_id_file)


if __name__ == '__main__':
    main()
