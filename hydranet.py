#!/usr/bin/env python3
"""
Class for holding of all of the branches of network together. Managing training.
"""

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, MaxPooling2D,\
    UpSampling2D, Merge, LSTM, Flatten, ZeroPadding2D, Reshape, BatchNormalization, Dropout
from keras.layers.wrappers import TimeDistributed

import network_structure


class HydraNet(object):
    def __init__(self, **kwargs):
        # self.models_folder = kwargs.get('models_folder', MODELS_FOLDER)

        self.structure = kwargs.get('structure', network_params.DEFAULT_STRUCTURE)

        self.pred_ahead = True

        # branches of network
        self.encoder = None
        self.decoder = None
        self.generator = None
        self.state_predictor = None
        self.action_mapper = None
        self.action_predictor = None
        self.state_sampler = None

        self.discriminator = None
        self.screen_discriminator = None
        self.state_discriminator = None

        # full networks
        self.autoencoder = None
        self.autoencoder_gen = None
        self.autoencoder_disc = None
        self.autoencoder_gan = None

        self.screen_predictor = None
        self.screen_predictor_g = None
        self.screen_predictor_d = None

        self.state_assigner = None
        self.future_sampler_g = None
        self.future_sampler_d = None

        self.build_branches()
        self.build_networks()