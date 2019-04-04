# -*- coding: utf-8 -*-

####################### 测试 ############################
import torch
from torchvision.models.resnet import BasicBlock
from PIL import Image

if __name__ == '__main__':
    img_path = "../../data/flickr8k/synthetic_image/val_0.jpg"
    img = Image.open(img_path).convert('RGB')
    print img
