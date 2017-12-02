#!/usr/bin/env python3

import torch
import torch.nn as nn

IM_CHANNELS = 1
IM_WIDTH = 28
V_SIZE = 256


class Encoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(Encoder, self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(IM_CHANNELS, 16, 5, 2, 0),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 8, 3, 2, 0),
            # nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.fc_seq = nn.Sequential(
            nn.Linear(392, v_size)
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),




        def forward(x):
            h = self.nn_seq(x)
            h.view(h.size(0), -1)

# build encoder
input_im = Input(shape=IM_SHAPE)
h = Convolution2D(16, 5, 5, subsample=(2, 2), activation='relu', border_mode='same')(input_im)
h = Convolution2D(8, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(h)
h = Reshape((392,))(h)
v = Dense(self.v_size, activation='relu')(h)

m = Model(input_im, v, name='encoder')
draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
if self.verbose:
    m.summary()
self.encoder = m


class _netG(nn.Module):
    def __init__(self, z_size):
        super(_netG, self).__init__()
        self.nn_seq = nn.Sequential(
            nn.ConvTranspose2d(z_size, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
        )