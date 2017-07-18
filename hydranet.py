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

from keras.utils.visualize_util import plot as draw_network

import numpy as np
from tqdm import tqdm, trange
import time

import tensorflow as tf

from structured_container import DataContainer

FOLDER_DIAGRAMS = 'diagrams'
FOLDER_MODELS = 'models'

IM_WIDTH = 28
IM_HEIGHT = 28
IM_CHANNELS = 1
IM_SHAPE = (IM_WIDTH, IM_HEIGHT, IM_CHANNELS)

SERIES_SHIFT = 1
EP_LEN = 100 - SERIES_SHIFT

V_SIZE = 16

BATCH_SIZE = 32

DEFAULT_SCHEME = {
    'clear_training': True,  # run training on clear episodes (non-masked percepts)
    'clear_training_batches': 500,  # how many batches?
    'guaranteed_percepts': 4,  # how many first percepts are guaranteed to be non-masked?
    'uncertain_percepts': 8,  # how many further have a high chance to be non-masked?
    'p_levels': np.sqrt(np.linspace(0.05, 0.99, 10)).tolist(),  # progressing probabilities of masking percepts
    'p_level_batches': 200,  # how many batches per level
    'p_final': 0.99,  # final probability level
    'lr_final': 0.0002,
    'final_batches': 2000,  # number of batches for final training
}


class HydraNet(object):
    def __init__(self, **kwargs):
        self.training_scheme = kwargs.get('training_scheme', DEFAULT_SCHEME)
        self.clear_training = self.training_scheme.get('clear_training', True)
        self.clear_training_batches = self.training_scheme.get('clear_training_batches', 500)
        self.guaranteed_percepts = self.training_scheme.get('guaranteed_percepts', 4)
        self.uncertain_percepts = self.training_scheme.get('uncertain_percepts', 8)
        self.p_levels = self.training_scheme.get('p_levels', [])
        self.p_level_batches = self.training_scheme.get('p_level_batches', 200)
        self.p_final = self.training_scheme.get('p_final', None)
        self.lr_final_ = self.training_scheme.get('lr_final', 0.0002)
        self.final_batches = self.training_scheme.get('final_batches', 1000)

        self.v_size = kwargs.get('v_size', V_SIZE)

        # modules of network
        self.encoder = None
        self.decoder = None
        self.lstm_train = None
        self.lstm_replay = None

        # lstm_replay purposely skipped (weights are copied from lstm_train
        self.modules = [self.encoder, self.decoder, self.lstm_train]

        # self.state_sampler = None

        # full networks
        self.pred_ae = None
        self.stepper = None

        # data containers
        # self.train_box = DataContainer('data-balls/balls-train.pt', batch_size=32, ep_len_read=EP_LEN)
        # self.test_box = DataContainer('data-balls/balls-valid.pt', batch_size=32, ep_len_read=EP_LEN)

        # initialisation
        self.build_modules()
        # self.load_modules()
        self.build_heads()

    def build_modules(self):
        # build encoder
        input_im = Input(shape=IM_SHAPE)
        h = Convolution2D(16, 5, 5, subsample=(2, 2), activation='relu', border_mode='same')(input_im)
        h = Convolution2D(8, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(h)
        h = Reshape((392,))(h)
        v = Dense(self.v_size, activation='relu')(h)
        m = Model(input_im, v, name='encoder')

        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name))
        m.summary()
        self.encoder = m

        # build decoder
        input_v = Input(shape=(self.v_size,))
        h = Dense(8 * 7 * 7, activation='relu')(input_v)
        h = Reshape((7, 7, 8))(h)
        h = UpSampling2D((2, 2))(h)
        h = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(h)
        h = UpSampling2D((2, 2))(h)
        output_im = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(h)
        m = Model(input_v, output_im, name='decoder')

        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name))
        m.summary()
        self.decoder = m

        # lstm for training
        self.lstm_train = LSTM(V_SIZE, return_sequences=True)

        # lstm for replays
        self.lstm_replay = LSTM(V_SIZE, stateful=True)

        # TODO: aux variables: loss, position, velocity
        # TODO: generator (decoder with noise)

    def load_modules(self, folder=FOLDER_MODELS, tag='0'):
        for module in self.modules:
            fpath = '{}/{}-{}.hdf5'.format(folder, module.name, tag)
            module.load_weights(fpath)

        self.lstm_replay.set_weights(self.lstm_train.get_weights())

    def save_modules(self, folder=FOLDER_MODELS, tag='0'):
        for module in self.modules:
            fpath = '{}/{}-{}.hdf5'.format(folder, module.name, tag)
            module.save_weights(fpath)

    def build_heads(self):
        # build predictive autoencoder
        input_ims = Input(shape=(EP_LEN - SERIES_SHIFT, IM_WIDTH, IM_HEIGHT, IM_CHANNELS))
        td1 = TimeDistributed(self.encoder,
                                input_shape=(EP_LEN - SERIES_SHIFT,
                                             IM_WIDTH,
                                             IM_HEIGHT,
                                             IM_CHANNELS))

        h = td1(input_ims)
        h = self.lstm_train(h)
        td2 = TimeDistributed(self.decoder, input_shape=((EP_LEN, V_SIZE)))
        output_preds = td2(h)
        m = Model(input_ims, output_preds, name='pred_ae_train')
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name))
        m.summary()
        m.compile(optimizer=Adam(lr=0.001), loss='mse')
        self.pred_ae = m

        # build replayer
        input_im = Input(batch_shape=(1, IM_WIDTH, IM_HEIGHT, IM_CHANNELS))

        h = self.encoder(input_im)
        h = Reshape((1, self.v_size))(h)
        h = self.lstm_replay(h)
        output_recon = self.decoder(h)
        m = Model(input_im, output_recon, name='stepper')
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name))
        m.summary()
        self.stepper = m

# --------------------- TRAINING -------------------------------

    def mask_percepts(self, images, p):
        images_masked = np.copy(images)
        for_removal = np.random.random(EP_LEN) < p
        clear_percepts = self.guaranteed_percepts + np.random.randint(0, self.uncertain_percepts)
        for_removal[0:clear_percepts] = False
        images_masked[:, for_removal] = 0
        return images

    def train_pred_ae(self, image_getter, p=0.0, test=False):
        images = image_getter()
        if p > 0.0:
            images_masked = self.mask_percepts(images, p)
        else:
            images_masked = images

        if test:
            loss = self.pred_ae.test_on_batch(images_masked[:, 0:-SERIES_SHIFT, ...],
                                                   images[:, SERIES_SHIFT:, ...])
        else:
            loss = self.pred_ae.train_on_batch(images_masked[:, 0:-SERIES_SHIFT, ...],
                                                    images[:, SERIES_SHIFT:, ...])
        return loss

    def execute_scheme(self, train_getter, test_getter):
        if self.clear_training:
            bar = trange(self.clear_training_batches)
            for i in bar:
                # self.train_pred_ae(train_getter, p=0.0)
                time.sleep(2)
                tqdm.write('hello at %i' % i)

        while len(self.p_levels) > 0:
            pass

        if self.p_final is not None:
            pass

if __name__ == '__main__':
    hydra = HydraNet()
    hydra.execute_scheme(0,0)



