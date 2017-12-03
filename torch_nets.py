#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


############## autoencoder for ball sim predictions ##################

IM_CHANNELS = 1
IM_WIDTH = 28
V_SIZE = 256
N_FILTERS = 16
EP_LEN = 100


class Encoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(Encoder, self).__init__()
        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1

            # size: (channels, 28, 28)
            nn.Conv2d(IM_CHANNELS, N_FILTERS, kernel_size=4, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),

            # size: (N_Filters, 16, 16)
            nn.Conv2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # size: (N_FILTERS, 8, 8)
        )
        self.fc_seq = nn.Sequential(
            nn.Linear(N_FILTERS * 8 * 8, v_size),
            nn.ReLU(inplace=True),

            # second fc layer?
            # nn.Linear(v_size, v_size),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.conv_seq(x)
        out = self.fc_seq(h.view(h.size(0), -1))
        return out


class Decoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(Decoder, self).__init__()
        self.fc_seq = nn.Sequential(
            # second fc layer?
            # nn.Linear(v_size, v_size),
            # nn.ReLU(inplace=True),

            nn.Linear(v_size, N_FILTERS * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1

            # size: (N_FILTERS, 8, 8)
            nn.ConvTranspose2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=2, bias=True),
            nn.ReLU(True),

            # size: (N_FILTERS, 16, 16)
            nn.ConvTranspose2d(N_FILTERS, IM_CHANNELS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),

            # size: (N_FILTERS, 8, 8)
        )

    def forward(self, x):
        h = self.fc_seq(x)
        out = self.conv_seq(h.view(h.size(0), N_FILTERS, 8, 8))
        return out


class Autoencoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(v_size)
        self.decoder = Decoder(v_size)

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out


class PredictiveAutoencoder(nn.Module):
    def __init__(self, v_size=V_SIZE):
        super(PredictiveAutoencoder, self).__init__()
        self.encoder = Encoder(v_size)
        self.gru = nn.GRU(v_size, v_size, num_layers=1)
        self.decoder = Decoder(v_size)
        self.bn = nn.BatchNorm1d(v_size)

    def forward(self, x):
        ep_len = x.size(0)
        batch_size = x.size(1)

        h = self.encoder(x.view(ep_len * batch_size, x.size(2), x.size(3), x.size(4)))
        h, state_f = self.gru(h.view(ep_len, batch_size, -1))
        # h = h.view(ep_len, batch_size, -1)
        out = self.decoder(h.view(ep_len * batch_size, -1))
        # out = self.decoder(self.bn(h.view(ep_len * batch_size, -1)))
        # out = self.decoder(self.bn(h.view(ep_len * batch_size, -1)))

        return out.view(x.size())


if __name__ == "__main__":
    BATCH_SIZE = 32

    # net = Encoder()
    # x = Variable(torch.randn(BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    # net = Decoder()
    # x = Variable(torch.randn(BATCH_SIZE, V_SIZE))

    # net = Autoencoder()
    # x = Variable(torch.randn(BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    net = PredictiveAutoencoder()
    x = Variable(torch.randn(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    res = net(x)
    print(res)





    #
#
# ############## discriminator for mnist/databalls ##################
# # out_width = (in_width - filter_width + 2*paddin)/stride + 1
#
# IM_CHANNELS = 1
# IM_WIDTH = 28
# # V_SIZE = 256
# N_FILTERS = 16
#
# class Discriminator28(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator28, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # size: (channels, 28, 28)
#             nn.Conv2d(IM_CHANNELS, N_FILTERS, 4, 2, 3, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # size: (N_Filters, 16, 16)
#             nn.Conv2d(N_FILTERS, N_FILTERS * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(N_FILTERS * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # size: (N_Filters, 8, 8)
#             nn.Conv2d(N_FILTERS * 2, N_FILTERS * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(N_FILTERS * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # size: (N_Filters, 4, 4)
#             nn.Conv2d(N_FILTERS * 4, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#
#             # size: (N_Filters, 1, 1)
#         )
#
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#
#         return output.view(-1, 1)