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

import tensorflow as tf

from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import PIL
import numpy as np
import time
import imageio
from tqdm import tqdm, trange

from structured_container import DataContainer
from particle_filter import ParticleFilter

FOLDER_DIAGRAMS = 'diagrams'
FOLDER_MODELS = 'models'
FOLDER_PLOTS = 'plots'
HEADER_HEIGHT = 18

IM_WIDTH = 28
IM_HEIGHT = 28
IM_CHANNELS = 1
IM_SHAPE = (IM_WIDTH, IM_HEIGHT, IM_CHANNELS)

SERIES_SHIFT = 1
EP_LEN = 100 - SERIES_SHIFT

V_SIZE = 64

BATCH_SIZE = 32
TEST_EVERY_N_BATCHES = 10

DEFAULT_SCHEME = {
    'clear_training': True,  # run training on clear episodes (non-masked percepts)
    # 'clear_training': False,  # run training on clear episodes (non-masked percepts)
    'clear_batches': 500,  # how many batches?
    'lr_initial': 0.001,
    'guaranteed_percepts': 4,  # how many first percepts are guaranteed to be non-masked?
    'uncertain_percepts': 8,  # how many further have a high chance to be non-masked?
    'p_levels': np.sqrt(np.linspace(0.05, 0.99, 10)).tolist(),  # progressing probabilities of masking percepts
    # 'p_levels': [],  # progressing probabilities of masking percepts
    'p_level_batches': 400,  # how many batches per level
    'p_final': 0.99,  # final probability level
    'lr_final': 0.0002,
    'final_batches': 4000,  # number of batches for final training
}


def create_im_label(label):
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", HEADER_HEIGHT)
    im = Image.new('F', (IM_WIDTH, HEADER_HEIGHT), 1.0)
    draw = ImageDraw.Draw(im)
    draw.text((1, 0), label, 0.0, font)

    return np.array(im)


class HydraNet(object):
    def __init__(self, **kwargs):
        # training configuration
        self.training_scheme = DEFAULT_SCHEME
        self.training_scheme.update(kwargs)

        self.clear_training = self.training_scheme.get('clear_training')
        self.clear_batches = self.training_scheme.get('clear_batches')
        self.lr_initial = self.training_scheme.get('lr_initial')
        self.guaranteed_percepts = self.training_scheme.get('guaranteed_percepts')
        self.uncertain_percepts = self.training_scheme.get('uncertain_percepts')
        self.p_levels = self.training_scheme.get('p_levels')
        self.p_level_batches = self.training_scheme.get('p_level_batches')
        self.p_final = self.training_scheme.get('p_final')
        self.lr_final = self.training_scheme.get('lr_final')
        self.final_batches = self.training_scheme.get('final_batches')

        # loss trackers
        self.pred_loss_train = []
        self.pred_loss_test = []

        # network configuration
        self.v_size = kwargs.get('v_size', V_SIZE)

        # modules of network
        self.encoder = None
        self.decoder = None
        self.state_pred_train = None

        # special layers
        self.lstm_replay = None

        # full networks
        self.pred_ae = None
        self.stepper = None

        # build network
        self.build_modules()
        # self.load_modules()
        self.build_heads()

        # lstm_replay purposely skipped (weights are copied from lstm_train)
        self.modules = [self.encoder, self.decoder, self.state_pred_train]

    def build_modules(self):
        # build encoder
        input_im = Input(shape=IM_SHAPE)
        h = Convolution2D(16, 5, 5, subsample=(2, 2), activation='relu', border_mode='same')(input_im)
        h = Convolution2D(8, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(h)
        h = Reshape((392,))(h)
        v = Dense(self.v_size, activation='relu')(h)

        m = Model(input_im, v, name='encoder')
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
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
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
        m.summary()
        self.decoder = m

        # lstm for training
        input_vs = Input(shape=(EP_LEN - SERIES_SHIFT, self.v_size,))
        output_vs = LSTM(V_SIZE, return_sequences=True)(input_vs)

        m = Model(input_vs, output_vs, name='state_pred_train')
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
        m.summary()
        self.state_pred_train = m

        # lstm for replays
        self.lstm_replay = LSTM(V_SIZE, stateful=True)

        # TODO: aux variables: loss, position, velocity
        # TODO: generator (decoder with noise)

    def load_modules(self, folder=FOLDER_MODELS, tag='0'):
        for module in self.modules:
            fpath = '{}/{}-{}.hdf5'.format(folder, module.name, tag)
            print('Loading {} from {}'.format(module.name, fpath))
            module.load_weights(fpath)

        self.lstm_replay.set_weights(self.state_pred_train.get_weights())

    def save_modules(self, folder=FOLDER_MODELS, tag='0'):
        for module in self.modules:
            fpath = '{}/{}-{}.hdf5'.format(folder, module.name, tag)
            print('Saving {} to {}'.format(module.name, fpath))
            module.save_weights(fpath)

    def load_model(self, fpath):
        self.pred_ae.load_weights(fpath)
        self.lstm_replay.set_weights(self.state_pred_train.get_weights())

    def build_heads(self):
        # build predictive autoencoder
        input_ims = Input(shape=(EP_LEN - SERIES_SHIFT, IM_WIDTH, IM_HEIGHT, IM_CHANNELS))
        td1 = TimeDistributed(self.encoder,
                                input_shape=(EP_LEN - SERIES_SHIFT,
                                             IM_WIDTH,
                                             IM_HEIGHT,
                                             IM_CHANNELS))

        h = td1(input_ims)
        h = self.state_pred_train(h)
        td2 = TimeDistributed(self.decoder, input_shape=(EP_LEN, self.v_size))
        output_preds = td2(h)
        m = Model(input_ims, output_preds, name='pred_ae_train')
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
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
        draw_network(m, to_file='{0}/{1}.png'.format(FOLDER_DIAGRAMS, m.name), show_layer_names=True, show_shapes=True)
        m.summary()
        self.stepper = m

# --------------------- TRAINING -------------------------------

    def mask_percepts(self, images, p):
        images_masked = np.copy(images)
        for_removal = np.random.random(EP_LEN) < p
        clear_percepts = self.guaranteed_percepts + np.random.randint(0, self.uncertain_percepts)
        for_removal[0:clear_percepts] = False
        images_masked[:, for_removal, ...] = 0
        return images_masked

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
        self.pred_ae.compile(optimizer=Adam(lr=self.lr_initial), loss='mse')

        postfix = {}
        if self.clear_training:
            current_p = 0.0
            print('Current p:', current_p)
            bar = trange(self.clear_batches)
            for i in bar:
                self.pred_loss_train.append(self.train_pred_ae(train_getter, p=current_p))
                smooth_loss = np.mean(self.pred_loss_train[-10:])
                postfix['L train'] = smooth_loss

                if i % TEST_EVERY_N_BATCHES == 0:
                    self.pred_loss_test.append(self.train_pred_ae(test_getter, p=current_p, test=True))
                    smooth_loss = np.mean(self.pred_loss_test[-4:])
                    postfix['L test'] = smooth_loss

                bar.set_postfix(**postfix)

            self.save_modules()

        while len(self.p_levels) > 0:
            current_p = self.p_levels.pop(0)

            print('Current p:', current_p)
            bar = trange(self.p_level_batches)
            for i in bar:
                self.pred_loss_train.append(self.train_pred_ae(train_getter, p=current_p))
                smooth_loss = np.mean(self.pred_loss_train[-10:])
                postfix['L train'] = smooth_loss

                if i % TEST_EVERY_N_BATCHES == 0:
                    self.pred_loss_test.append(self.train_pred_ae(test_getter, p=current_p, test=True))
                    smooth_loss = np.mean(self.pred_loss_test[-4:])
                    postfix['L test'] = smooth_loss

                bar.set_postfix(**postfix)

        if self.p_final is not None:
            self.pred_ae.compile(optimizer=Adam(lr=self.lr_final), loss='mse')

            current_p = self.p_final
            print('Current p:', current_p)
            bar = trange(self.final_batches)
            for i in bar:
                self.pred_loss_train.append(self.train_pred_ae(train_getter, p=current_p))
                smooth_loss = np.mean(self.pred_loss_train[-10:])
                postfix['L train'] = smooth_loss

                if i % TEST_EVERY_N_BATCHES == 0:
                    self.pred_loss_test.append(self.train_pred_ae(test_getter, p=current_p, test=True))
                    smooth_loss = np.mean(self.pred_loss_test[-4:])
                    postfix['L test'] = smooth_loss

                bar.set_postfix(**postfix)

            self.save_modules()

# ------------------------- DISPLAYING --------------------------------

    def plot_losses(self, tag=0):
        plt.plot(self.pred_loss_train)
        batches = np.arange(len(self.pred_loss_test)) * TEST_EVERY_N_BATCHES
        plt.plot(self.pred_loss_train, batches)
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('updates')
        plt.legend(['train', 'test'])
        fpath = '{}/loss-{}.gif'.format(FOLDER_PLOTS, tag)
        plt.savefig(fpath)

    def draw_pred_gif(self, test_getter, p=1.0, use_pf=False, sim_config=None, use_stepper=False, tag=0):
        episode = test_getter()
        episode = episode[0, ...].reshape((1,) + episode.shape[1:])
        episode_masked = self.mask_percepts(episode, p)
        net_preds = self.pred_ae.predict(episode_masked[:, 0:-SERIES_SHIFT, ...])

        # stepper predictions
        stepper_pred = []
        if use_stepper:
            self.stepper.reset_states()
            for t in range(EP_LEN-SERIES_SHIFT):
                im = episode_masked[:, t, ...]
                pred = self.stepper.predict(im)
                stepper_pred.append(self.stepper.predict(im))

        pf_pred = []
        if use_pf:
            pf = ParticleFilter(sim_config)
            for t in range(EP_LEN-SERIES_SHIFT):
                im = episode_masked[:, t, ...]
                pass

        # create header with labels
        col = np.zeros((HEADER_HEIGHT, 1))

        labels = []
        labels.append(create_im_label('GT'))
        labels.append(create_im_label('AE'))
        if use_stepper:
            labels.append(create_im_label('ST'))
        if use_pf:
            labels.append(create_im_label('PF'))

        header = [col]
        for label in labels:
            header.append(label)
            header.append(col)

        header = np.concatenate(header, axis=1)

        # combine predictions
        col = np.ones((IM_HEIGHT, 1))
        frames = []
        for t in range(EP_LEN - SERIES_SHIFT):
            images = []
            images.append(episode[0, t+SERIES_SHIFT, :, :, 0])
            images.append(net_preds[0, t, :, :, 0])
            if use_stepper:
                images.append(stepper_pred[t][0, :, :, 0])

            if use_pf:
                images.append(pf_pred[t])

            table = [col]
            for image in images:
                table.append(image)
                table.append(col)

            frame = np.concatenate(table, axis=1)

            # print(frame.shape)
            width = frame.shape[1]
            row = np.ones((1, width))
            frame = np.concatenate([header, frame, row], axis=0)

            frames.append(frame)

        fpath = '{}/predictions-{}.gif'.format(FOLDER_PLOTS, tag)
        imageio.mimsave(fpath, frames)


if __name__ == '__main__':
    train_box = DataContainer('data-balls/balls-train.pt', batch_size=32, ep_len_read=EP_LEN)
    test_box = DataContainer('data-balls/balls-valid.pt', batch_size=32, ep_len_read=EP_LEN)
    train_box.populate_images()
    test_box.populate_images()

    hydra = HydraNet()
    # hydra.load_modules()
    hydra.execute_scheme(train_box.get_batch_episodes, test_box.get_batch_episodes)



