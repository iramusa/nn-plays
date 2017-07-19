#!/usr/bin/env python3

import numpy as np
import random
import torch

from balls_sim import WORLD_LEN


class DataContainer(object):
    def __init__(self, file, batch_size, ep_len_read=20, shape=((28, 28, 1))):
        self.file = file
        self.ep_len_read = ep_len_read
        self.batch_size = batch_size

        self.sim_config = None

        self.n_episodes = None
        self.record = None
        self.images = None
        self.images_populated = False

        self.load_record(file)

        self.im_shape = shape

        space = np.linspace(0.5, WORLD_LEN-0.5, WORLD_LEN)
        self.I, self.J = np.meshgrid(space, space)

    def set_ep_len(self, ep_len):
        self.ep_len_read = ep_len

    def load_record(self, file):
        self.record = torch.load(file)
        self.n_episodes = self.record['n_episodes']
        self.sim_config = self.record['sim_config']

    def populate_images(self):
        if self.images_populated:
            return
        else:
            self.images = np.zeros((self.n_episodes, self.ep_len_read) + self.im_shape)
            for i, episode in enumerate(self.record['episodes']):
                self.images[i, ...] = self.episode2images(episode)

    def destroy_images(self):
        self.images = None

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

                # for non rolling images simply:
                # images[i, :, :, 0] += np.exp(-(((self.I - pos_x) ** 2 + (self.J - pos_y) ** 2) / (radii[j] ** 2)) ** 4)

                x_dist = (np.abs(self.I - pos_x) + WORLD_LEN/2) % WORLD_LEN - WORLD_LEN/2
                y_dist = (np.abs(self.J - pos_y) + WORLD_LEN/2) % WORLD_LEN - WORLD_LEN/2
                images[i, :, :, 0] += np.exp(-((x_dist ** 2 + y_dist ** 2) / (radii[j] ** 2)) ** 4)

        images = np.clip(images, 0, 1)
        return images

    def get_n_random_episodes(self, n):
        if self.images is None:
            eps = self.get_n_random_structured_episodes(n)
            images = np.array([self.episode2images(ep) for ep in eps])

        else:
            ep_rolls = np.random.randint(0, self.n_episodes, n)
            images = self.images[ep_rolls, ...].astype('float')

            return images

        return images

    def get_n_random_episodes_full(self, n=2):
        eps = self.get_n_random_structured_episodes(n)
        images = np.array([self.episode2images(ep) for ep in eps])

        eps_poses = []
        for ep in eps:
            poses = []
            for t in ep['t_list']:
                poses.append(t['poses'])

            eps_poses.append(poses)

        poses = np.array(eps_poses)
        return images, poses

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
