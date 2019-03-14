# -*- coding: utf-8 -*-


# ############################## For Extract Image Feature Representations ##############################
# we use ResNet to extract image feature representations with pre-trained model
# 使用预训练的残差网络提取图像特征
# reference
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

import torch
import torch.nn as nn
from torchvision.models import resnet152


class DownsampleNetwork(nn.Module):

    def __init__(self, arguments):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(DownsampleNetwork, self).__init__()
        self.image_feature_size = arguments['image_feature_size']
        resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, self.image_feature_size)
        self.bn = nn.BatchNorm1d(self.image_feature_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
