# -*- coding: utf-8 -*-

####################### 对特征进行上采样，生成图像 ############################
##### reference
# https://github.com/aelnouby/Text-to-Image-Synthesis


import torch
import torch.nn as nn


class UpsampleNetwork(nn.Module):
    def __init__(self, arguments):
        super(UpsampleNetwork, self).__init__()
        self.ngf = arguments['ngf']
        self.input_size = arguments['up_sample_input_dim']
        self.num_channels = arguments['num_channels']

        self.generator = nn.Sequential(
            # input (self.noise_dim + self.sentence_embedding_size)*(1*1)
            nn.ConvTranspose2d(self.input_size, self.ngf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.ngf, self.ngf / 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf / 2),
            nn.Tanh(),

            # state size. (ngf/2) x 128 x 128
            nn.ConvTranspose2d(self.ngf / 2, self.num_channels, kernel_size=2, stride=2, padding=16, bias=False),
            nn.Tanh()
            # state size. num_channels x 224 x 224
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, sentence_embedding, noise=None):
        if noise is not None:
            input = torch.cat((sentence_embedding, noise), 1)
        else:
            input = sentence_embedding

        output = self.generator(input)
        return output
