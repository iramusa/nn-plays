#!/usr/bin/env python3

import numpy as np
import torch


class DataContainer(object):
    def __init__(self, file, batch_size, ep_len_read=20, episodes=200, shape=((28, 28, 1))):
        self.file = file
        self.ep_len_read = ep_len_read
        self.ep_len_gen = ep_len_read
        self.episodes = episodes
        self.total_images = self.ep_len_read * self.episodes
        self.batch_size = batch_size
        # self.im_med = np.zeros(self.im_shape)
        self.images = None

        self.load_images(file)
        # self.im_shape = self.images[0, 0].shape
        self.im_shape = shape

        # TODO get a median image, not episode
        # self.im_med = np.median(self.images, axis=0)[0]

        # del self.single_image
        # self.sess.close()
        # mean based shift
        # self.im_med = np.mean(self.images, axis=0) * (1.0/255.0)

    def set_ep_len(self, ep_len):
        self.ep_len_gen = ep_len

    def load_images(self, file):
        self.images = torch.load(file)

    def get_n_random_images(self, n):
        ep_rolls = np.random.randint(0, self.episodes, n)
        t_rolls = np.random.randint(0, self.ep_len_read, n)

        random_ims = self.images[ep_rolls, t_rolls, ...]
        return random_ims.reshape((n,) + self.im_shape)

    def get_batch_images(self):
        return self.get_n_random_images(self.batch_size).astype('float')

    def get_n_random_episodes(self, n, ep_len=None):
        if ep_len is None:
            ep_len = self.ep_len_gen

        ep_rolls = np.random.randint(0, self.episodes, n)
        random_eps = self.images[ep_rolls, :ep_len, ...].astype('float')

        return random_eps.reshape((n, ep_len) + self.im_shape)

    def get_episode(self):
        return self.get_n_random_episodes(1)[0]

    def get_batch_episodes(self):
        return self.get_n_random_episodes(self.batch_size)

    def get_n_batches_images(self, n=10):
        ims = []
        for i in range(n):
            im = self.get_batch_images()
            ims.append(im)

        ims = np.concatenate(ims, axis=0)

        return ims

    def generate_ae(self):
        while True:
            images = self.get_batch_images()
            yield (images, images)

    def generate_ae_gan(self):
        while True:
            images = self.get_batch_images()
            labels = np.ones((images.shape[0],))
            yield (images, labels)

    def generate_ae_gan_mo(self):
        while True:
            images = self.get_batch_images()
            labels = np.ones((images.shape[0],))
            yield (images, [images, labels])

