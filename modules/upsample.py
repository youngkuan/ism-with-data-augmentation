# -*- coding: utf-8 -*-

####################### 对特征进行上采样，生成图像 ############################
##### reference


import torch.nn as nn
from upsample_text2image import UpsampleNetwork as UpsampleNetworkT2I


class UpsampleNetwork(nn.Module):
    def __init__(self, arguments):
        super(UpsampleNetwork, self).__init__()
        self.ngf = arguments['ngf']
        self.input_dim = arguments['up_sample_input_dim']
        # self.define_module()
        self.upsampleNetworkT2I = UpsampleNetworkT2I(arguments)

    def forward(self, sentence_embedding, noise=None):
        return self.upsampleNetworkT2I(sentence_embedding, noise)
