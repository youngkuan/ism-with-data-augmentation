# -*- coding: utf-8 -*-

####################### 对特征进行上采样，生成图像 ############################
##### reference
# https://github.com/hanzhanggit/StackGAN-Pytorch


import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class UpsampleNetwork(nn.Module):
    def __init__(self, arguments):
        super(UpsampleNetwork, self).__init__()
        self.ngf = arguments['ngf']
        self.input_dim = arguments['up_sample_input_dim']
        self.define_module()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = self.upsample(self.ngf, self.ngf // 2)
        self.upsample2 = self.upsample(self.ngf // 2, self.ngf // 4)
        self.upsample3 = self.upsample(self.ngf // 4, self.ngf // 8)
        self.upsample4 = self.upsample(self.ngf // 8, self.ngf // 16)

    def conv3x3(self, in_channels, out_channels):
        """
        3x3 convolution with padding
        :param in_channels:
        :param out_channels:
        :return:
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                         padding=1, bias=False)

    def upsample(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            self.conv3x3(in_channels, out_channels * 2),
            nn.BatchNorm2d(out_channels * 2),
            self.GLU()
        )
        return block

    def forward(self, sentence_embedding, noise=None):
        if noise is not None:
            input = torch.cat((sentence_embedding, noise), 1)
        else:
            input = sentence_embedding

        # state size 16ngf x 4 x 4
        output = self.fc(input)
        output = output.view(-1, self.ngf, 4, 4)
        # state size 8ngf x 8 x 8
        output = self.upsample1(output)
        # state size 4ngf x 16 x 16
        output = self.upsample2(output)
        # state size 2ngf x 32 x 32
        output = self.upsample3(output)
        # state size ngf x 64 x 64
        output = self.upsample4(output)

        return output
