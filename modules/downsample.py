# -*- coding: utf-8 -*-


# ############################## For Extract Image Feature Representations ##############################
# we use ResNet to extract image feature representations with pre-trained model
# 使用预训练的残差网络提取图像特征

import torch.nn as nn
from torchvision.models import resnet101


class DownsampleNetwork(nn.Module):

    def __init__(self, arguments):
        ## 使用残差网络提取图像特征
        self.encoder = resnet101(pretrained=True)

    def forward(self, image):
        image_feature = self.encoder(image)

        return image_feature
