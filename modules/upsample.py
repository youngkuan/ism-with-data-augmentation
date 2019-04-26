# -*- coding: utf-8 -*-

####################### 对特征进行上采样，生成图像 ############################
##### reference
# https://github.com/hanzhanggit/StackGAN-Pytorch


import torch
import torch.nn as nn


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, arguments):
        super(CA_NET, self).__init__()
        self.t_dim = arguments["embed_size"]
        self.c_dim = arguments["condition_dimension"]
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class UpsampleNetwork(nn.Module):
    def __init__(self, arguments):
        super(UpsampleNetwork, self).__init__()
        self.ngf = arguments['ngf']
        self.input_dim = arguments['noise_dim'] + arguments["condition_dimension"]

        # SENTENCE.DIMENSION -> GAN.CONDITION_DIMENSION
        self.ca_net = CA_NET(arguments)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(self.ngf * 4 * 4),
            nn.ReLU(True))

        self.upsample1 = self.upsample(self.ngf, self.ngf // 2)
        self.upsample2 = self.upsample(self.ngf // 2, self.ngf // 4)
        self.upsample3 = self.upsample(self.ngf // 4, self.ngf // 8)
        self.upsample4 = self.upsample(self.ngf // 8, self.ngf // 16)
        # self.upsample5 = self.upsample(self.ngf // 16, self.ngf // 32)
        # self.upsample6 = self.upsample(self.ngf // 32, self.ngf // 64)
        #
        # self.reshape = nn.Conv2d(self.ngf // 64, self.ngf // 64, kernel_size=9, dilation=4)

        self.img = nn.Sequential(
            self.conv3x3(self.ngf // 16, 3),
            nn.Tanh())

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
            self.conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        return block

    def forward(self, sentence_embedding, noise):
        c_code, mu, logvar = self.ca_net(sentence_embedding)
        z_c_code = torch.cat((noise, c_code), 1)

        # state size ngf x 4 x 4
        output = self.fc(z_c_code)
        output = output.view(-1, self.ngf, 4, 4)
        # state size ngf/2 x 8 x 8
        output = self.upsample1(output)
        # state size ngf/4 x 16 x 16
        output = self.upsample2(output)
        # state size ngf/8 x 32 x 32
        output = self.upsample3(output)
        # state size ngf/16 x 64 x 64
        output = self.upsample4(output)

        # output = self.upsample5(output)
        # # -> upsample5: self.ngf/32 * (128*128)
        # output = self.upsample6(output)
        # # -> upsample6: self.ngf/64 * (256*256)
        # output = self.reshape(output)
        # # -> reshape: self.ngf/64 * (224*224)

        # state size 3 x 224 x 224
        fake_img = self.img(output)
        return fake_img, mu, logvar
