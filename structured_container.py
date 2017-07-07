#!/usr/bin/env python3

import numpy as np
import random
import torch

import balls_sim


class DataContainer(object):
    def __init__(self, file, batch_size, ep_len_read=20, episodes=200, shape=((28, 28, 1))):
        self.file = file
        self.ep_len_read = ep_len_read
        self.batch_size = batch_size

        self.n_episodes = None
        self.record = None

        self.load_record(file)

        self.im_shape = shape

        space = np.linspace(0.5, balls_sim.WORLD_LEN-0.5, balls_sim.WORLD_LEN)
        self.I, self.J = np.meshgrid(space, space)

    def set_ep_len(self, ep_len):
        self.ep_len_read = ep_len

    def load_record(self, file):
        self.record = torch.load(file)
        self.n_episodes = self.record['n_episodes']

    def get_n_random_structured_episodes(self, n):
        random_eps = random.sample(self.record['episodes'], n)
        return random_eps

    def episode2images(self, episode, noisy=False):
        radii = episode['radii']
        ts = episode['t_list']

        images = np.zeros((self.ep_len_read,) + self.im_shape)

        for i in range(self.ep_len_read):
            t = ts[i]
            if not noisy:
                points = t['poses']
            else:
                points = t['measures']

            for j, point in enumerate(points):
                pos_x = point[0]
                pos_y = point[1]

                images[i, :, :, 0] += np.exp(-(((self.I - pos_x) ** 2 + (self.J - pos_y) ** 2) / (radii[j] ** 2)) ** 4)

        images = np.clip(images, 0, 1)
        return images

    def get_n_random_episodes(self, n):
        eps = self.get_n_random_structured_episodes(n)
        images = np.array([self.episode2images(ep) for ep in eps])

        return images

    def get_episode(self):
        return self.get_n_random_episodes(1)[0]

    def get_batch_episodes(self):
        return self.get_n_random_episodes(self.batch_size)

    # def generate_ae(self):
    #     while True:
    #         images = self.get_batch_images()
    #         yield (images, images)
    #
    # def generate_ae_gan(self):
    #     while True:
    #         images = self.get_batch_images()
    #         labels = np.ones((images.shape[0],))
    #         yield (images, labels)
    #
    # def generate_ae_gan_mo(self):
    #     while True:
    #         images = self.get_batch_images()
    #         labels = np.ones((images.shape[0],))
    #         yield (images, [images, labels])
    #
