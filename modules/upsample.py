# -*- coding: utf-8 -*-

####################### 对特征进行上采样，生成图像 ############################
##### reference


import torch.nn as nn

from upsample_stack_gan import UpsampleNetwork as UpsampleNetworkSG
from utils import weights_init


class UpsampleNetwork(nn.Module):
    def __init__(self, arguments):
        super(UpsampleNetwork, self).__init__()
        # self.define_module()
        # self.upsampleNetworkT2I = UpsampleNetworkT2I(arguments)
        self.upsampleNetworkSG = UpsampleNetworkSG(arguments)
        self.upsampleNetworkSG.apply(weights_init)

    def forward(self, sentence_embedding, noise):
        # return self.upsampleNetworkT2I(sentence_embedding, noise)
        return self.upsampleNetworkSG(sentence_embedding, noise)
