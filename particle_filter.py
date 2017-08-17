#!/usr/bin/env python3

import numpy as np
import random
import copy

import balls_sim
from balls_sim import default_config, WORLD_LEN


def norm_pdf(x):
    # true
    # return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    # multiplied by a constant
    return np.exp(-x ** 2)


class ParticleFilter(object):
    def __init__(self, sim_config=default_config, n_particles=10000):
        self.sim_config = default_config
        self.sim_config.update(sim_config)
        self.n = n_particles
        self.parts = []

        self.n_targets = sim_config['n_bodies']

        self.measurement_noise = sim_config['measurement_noise']
        if self.measurement_noise == 0.0:
            self.measurement_noise = 0.05

        self.dynamics_noise = sim_config['dynamics_noise']
        if self.dynamics_noise == 0.0:
            self.dynamics_noise = 0.1

        for _ in range(self.n):
            self.parts.append(balls_sim.World(**sim_config))
        self.w = np.ones(n_particles)/n_particles

        space = np.linspace(0.5, WORLD_LEN - 0.5, WORLD_LEN)
        self.I, self.J = np.meshgrid(space, space)

    def add_noise(self, noise_level=0.2):
        for part in self.parts:
            for body in part.bodies:
                body.vel += noise_level * np.random.randn(2)

    def predict(self):
        for part in self.parts:
            part.run()

    def update(self, measurement):
        for i, part in enumerate(self.parts):
            if self.n_targets == 1:
                # circular noise, do rectangular?
                dist = np.linalg.norm(measurement - part.bodies[0].pos)
                self.w[i] *= norm_pdf(dist/self.measurement_noise)
            else:
                raise ValueError('Wrong number of targets')

        self.w /= np.sum(self.w)

    def resample(self):
        indices = np.array(range(self.n))
        samples_i = np.random.choice(indices, self.n, p=self.w)

        new_parts = []
        for i in range(self.n):
            new_parts.append(copy.deepcopy(self.parts[samples_i[i]]))

        self.parts = new_parts
        self.w = np.ones(self.n)/self.n

    def get_distributions(self):
        if self.n_targets == 1:
            poses = [part.bodies[0].pos for part in self.parts]
            vels = [part.bodies[0].vel for part in self.parts]

            poses = np.array(poses)
            vels = np.array(vels)
        else:
            raise ValueError('Wrong number of targets')

        return poses, vels

    def get_stats(self):
        if self.n_targets == 1:
            poses, vels = self.get_distributions()

            pos_mean = np.mean(poses, axis=0)
            vel_mean = np.mean(vels, axis=0)

            pos_std = np.sqrt(np.var(poses, axis=0))
            vel_std = np.sqrt(np.var(vels, axis=0))
        else:
            raise ValueError('Wrong number of targets')

        return pos_mean, pos_std, vel_mean, vel_std

    def draw(self):
        image = np.zeros((WORLD_LEN, WORLD_LEN, 1))

        for i, part in enumerate(self.parts):
            part_im = np.zeros((WORLD_LEN, WORLD_LEN))
            for body in part.bodies:
                pos_x = body.pos[0]
                pos_y = body.pos[1]
                radius = body.r

                # for non rolling images simply:
                # images[i, :, :, 0] += np.exp(-(((self.I - pos_x) ** 2 + (self.J - pos_y) ** 2) / (radii[j] ** 2)) ** 4)

                x_dist = (np.abs(self.I - pos_x) + WORLD_LEN/2) % WORLD_LEN - WORLD_LEN/2
                y_dist = (np.abs(self.J - pos_y) + WORLD_LEN/2) % WORLD_LEN - WORLD_LEN/2
                body_im = np.exp(-((x_dist ** 2 + y_dist ** 2) / (radius ** 2)) ** 4)
                part_im += body_im

            part_im = np.clip(part_im, 0, 1)
            image[:, :, 0] += part_im * self.w[i]

        return image
