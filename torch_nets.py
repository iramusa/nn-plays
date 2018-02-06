#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import torch.nn.functional as F


############## autoencoder for ball sim predictions ##################

IM_CHANNELS = 1
IM_WIDTH = 28

V_SIZE = 256
BS_SIZE = 256
N_SIZE = 256
D_SIZE = 64
G_SIZE = 256

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
            # nn.BatchNorm1d(v_size),
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
    def __init__(self, bs_size=BS_SIZE):
        super(Decoder, self).__init__()
        self.fc_seq = nn.Sequential(
            # second fc layer?
            # nn.Linear(v_size, v_size),
            # nn.ReLU(inplace=True),

            nn.Linear(bs_size, N_FILTERS * 8 * 8),
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


class BeliefStatePropagator(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE):
        super(BeliefStatePropagator, self).__init__()
        self.encoder = Encoder(v_size)
        self.gru = nn.GRU(v_size, bs_size, num_layers=1)

    def forward(self, x):
        ep_len = x.size(0)
        batch_size = x.size(1)

        h = self.encoder(x.view(ep_len * batch_size, x.size(2), x.size(3), x.size(4)))
        out, state_f = self.gru(h.view(ep_len, batch_size, -1))
        return out


class PredictiveAutoencoder(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE):
        super(PredictiveAutoencoder, self).__init__()
        self.bs_prop = BeliefStatePropagator(v_size, bs_size)
        self.decoder = Decoder(bs_size)
        # self.bn = nn.BatchNorm1d(v_size)

    def forward(self, x):
        ep_len = x.size(0)
        batch_size = x.size(1)

        h = self.bs_prop(x)
        out = self.decoder(h.view(ep_len * batch_size, -1))

        return out.view(x.size())


class BeliefStateDiscriminator(nn.Module):
    def __init__(self, bs_size=BS_SIZE, d_size=D_SIZE):
        super(BeliefStateDiscriminator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(bs_size, d_size),
            nn.ReLU(inplace=True),

            nn.Linear(d_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.fc_seq(x)
        return out


class BeliefStateGenerator(nn.Module):
    def __init__(self, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE):
        super(BeliefStateGenerator, self).__init__()
        self.fc_seq = nn.Sequential(
            nn.Linear(bs_size + n_size, g_size),
            nn.ReLU(inplace=True),

            nn.Linear(g_size, bs_size),
            nn.Tanh(),
        )

    def forward(self, noise, bs):
        """

        :param noise: should be normally distributed
        :param bs: from state distribution (somewhat similar to uniform)
        :return:
        """
        noise_joint = torch.cat([noise, bs], dim=-1)
        out = self.fc_seq(noise_joint)
        return out


class BeliefStateGAN(nn.Module):
    def __init__(self, bs_size=BS_SIZE, n_size=N_SIZE, d_size=D_SIZE, g_size=G_SIZE):
        super(BeliefStateGAN, self).__init__()
        self.D = BeliefStateDiscriminator(bs_size=bs_size, d_size=d_size)
        self.G = BeliefStateGenerator(bs_size=bs_size, n_size=n_size, g_size=g_size)

    def forward(self, noise, bs):
        """

        :param noise: should be normally distributed
        :param bs: from state distribution (somewhat similar to uniform)
        :return:
        """
        state = self.G(noise, bs)
        label = self.D(state)
        return label


class VisualDiscriminator(nn.Module):
    def __init__(self):
        super(VisualDiscriminator, self).__init__()
        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1

            # size: (channels, 28, 28)
            nn.Conv2d(IM_CHANNELS, N_FILTERS, kernel_size=4, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(N_FILTERS),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            # size: (N_Filters, 16, 16)
            nn.Conv2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(N_FILTERS),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            # size: (N_FILTERS, 8, 8)
        )
        self.fc_seq = nn.Sequential(
            nn.Linear(N_FILTERS * 8 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.conv_seq(x)
        out = self.fc_seq(h.view(h.size(0), -1))
        return out


class ConditionalVisualGenerator(nn.Module):
    def __init__(self, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE):
        super(ConditionalVisualGenerator, self).__init__()
        self.fc_seq = nn.Sequential(
            # nn.Linear(bs_size + n_size, g_size),
            # nn.BatchNorm1d(g_size),
            nn.Linear(bs_size + n_size, N_FILTERS * 8 * 8),
            nn.BatchNorm1d(N_FILTERS * 8 * 8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Linear(g_size, N_FILTERS * 8 * 8),
            # nn.BatchNorm1d(N_FILTERS * 8 * 8),
            # nn.ReLU(inplace=True),
        )
        self.conv_seq = nn.Sequential(
            # out_size = (in_size - kernel_size + 2*padding)/stride + 1

            # size: (N_FILTERS, 8, 8)
            nn.ConvTranspose2d(N_FILTERS, N_FILTERS, kernel_size=4, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(N_FILTERS),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            # size: (N_FILTERS, 16, 16)
            nn.ConvTranspose2d(N_FILTERS, IM_CHANNELS, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),

            # size: (N_FILTERS, 8, 8)
        )

    def forward(self, noise, bs):
        """

        :param noise: should be normally distributed
        :param bs: from state distribution (somewhat similar to uniform)
        :return:
        """
        noise_joint = torch.cat([noise, bs], dim=-1)
        h = self.fc_seq(noise_joint)
        out = self.conv_seq(h.view(h.size(0), N_FILTERS, 8, 8))
        return out


class VisualPAEGAN(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE):
        super(VisualPAEGAN, self).__init__()
        self.bs_prop = BeliefStatePropagator(v_size, bs_size)
        self.decoder = Decoder(bs_size)

        # version 1
        # self.D = VisualDiscriminator()
        # self.G = ConditionalVisualGenerator(bs_size=bs_size, n_size=n_size, g_size=g_size)

        # version stolen
        # self.D = StolenDiscriminator()
        # self.G = StolenGenerator()
        # self.D.weight_init(mean=0.0, std=0.02)
        # self.G.weight_init(mean=0.0, std=0.02)

        # version state gen
        self.D = StolenDiscriminator()
        self.G = BeliefStateGenerator()

    def forward(self):
        # ep_len = x.size(0)
        # batch_size = x.size(1)
        #
        # h = self.bs_prop(x)
        # obs_expectation = self.decoder(h.view(ep_len * batch_size, -1))
        # obs_sample = self.G(h.view(ep_len * batch_size, -1))
        #
        # return out.view(x.size())
        return None


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class StolenGenerator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(StolenGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(N_SIZE + BS_SIZE, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 2)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d*2, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise, input):
        noise_joint = torch.cat([noise, input], dim=-1)
        x = F.relu(self.deconv1_bn(self.deconv1(noise_joint.view(-1, N_SIZE + BS_SIZE, 1, 1))))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))

        return x


class StolenDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(StolenDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        # self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        # self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):

        x = F.leaky_relu(self.conv1(input), 0.2)
        # x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        x = F.sigmoid(self.conv5(x))

        return x


TS_RECON = 'time series reconstruction'
GAN = 'gan'
OBS_SAMPLE_AV = 'observable averaging of samples'
TS_SAMPLE_AV = 'full averaging of samples'


class PAEGAN(nn.Module):
    def __init__(self, v_size=V_SIZE, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE):
        super(PAEGAN, self).__init__()

        # POSSIBLE NETWORK COMPUTATIONS
        self.computations_switch = {
            TS_RECON: False,  # time series reconstruction
            GAN: False,  # observation generation
            OBS_SAMPLE_AV: False,  # averaging observable part of state samples
            TS_SAMPLE_AV: False,  # averaging entirety of state samples
        }

        self.encoder = Encoder(v_size)
        self.gru = nn.GRU(v_size, bs_size, num_layers=1)
        self.decoder = Decoder(bs_size)
        self.D = StolenDiscriminator()
        self.G = BeliefStateGenerator()

        # self.gan_noise = Variable(torch.FloatTensor(1, IM_CHANNELS, IM_WIDTH, IM_WIDTH))
        # self.averaging_noise = Variable(torch.FloatTensor(1, IM_CHANNELS, IM_WIDTH, IM_WIDTH))
        # self.null_image = Variable(torch.FloatTensor(1, IM_CHANNELS, IM_WIDTH, IM_WIDTH))
        # self.null_measurement

        if torch.cuda.is_available():
            self.null_image.cuda()
            self.cuda()


        #self.bs_prop = BeliefStatePropagator(v_size, bs_size)

        # version 1
        # self.D = VisualDiscriminator()
        # self.G = ConditionalVisualGenerator(bs_size=bs_size, n_size=n_size, g_size=g_size)

        # version stolen
        # self.D = StolenDiscriminator()
        # self.G = StolenGenerator()
        # self.D.weight_init(mean=0.0, std=0.02)
        # self.G.weight_init(mean=0.0, std=0.02)

    def propagate_states(self, ts):
        ep_len = ts.size(0)
        batch_size = ts.size(1)

        h = self.encoder(ts.view(ep_len * batch_size, ts.size(2), ts.size(3), ts.size(4)))
        states, state_f = self.gru(h.view(ep_len, batch_size, -1))
        return states

    def blind_propagate_states(self, state_0, n_timesteps):
        """
        Propagate belief states without sensory update
        :param state_0: initial states
        :param n_timesteps: for how many timesteps?
        :return: states: (n_timesteps, n_samples, bs_size)
        """
        # encoding of measurement which corresponds to null image
        null_encoding = self.encoder(self.null_image)
        if self.null_encoding.size(0) != n_timesteps:
            # self.null_encoding = torch.expand(null_encoding)
            pass

        if null_encoding != self.null_encoding[:,:, ...]:
            self.null_encoding[:, :, ...] = null_encoding

        states, state_f = self.gru(self.null_encoding)
        return states

    def visualise_states(self, states):
        return self.decoder(states)

    def ts_reconstruction(self, states, ep_len, batch_size):
        self.visualise_states(states).view(ep_len, batch_size, IM_CHANNELS, IM_WIDTH, IM_WIDTH)

    def sample_single_state(self, state, n_samples):
        pass

    def sample_many_states(self, states):
        pass

    def forward(self):
        # ep_len = x.size(0)
        # batch_size = x.size(1)
        #
        # h = self.bs_prop(x)
        # obs_expectation = self.decoder(h.view(ep_len * batch_size, -1))
        # obs_sample = self.G(h.view(ep_len * batch_size, -1))
        #
        # return out.view(x.size())
        return None


if __name__ == "__main__":
    BATCH_SIZE = 32

    # net = Encoder()
    # x = Variable(torch.randn(BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    # net = Decoder()
    # x = Variable(torch.randn(BATCH_SIZE, V_SIZE))

    # net = Autoencoder()
    # x = Variable(torch.randn(BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    # net = PredictiveAutoencoder()
    # x = Variable(torch.randn(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

    # net = BeliefStateGAN()
    # net = BeliefStateGenerator()
    G = StolenGenerator()
    D = StolenDiscriminator()
    noise = Variable(torch.randn(BATCH_SIZE, N_SIZE))
    bs = Variable(torch.randn(BATCH_SIZE, BS_SIZE))

    fake = G(noise, bs)
    print(fake)

    res = D(fake)
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