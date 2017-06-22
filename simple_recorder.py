#!/usr/bin/env python3
"""
Records images from simple sim
"""
import simple_sim
import balls_sim
import os
import numpy as np
import torch


# SIM = 'simple'
SIM = 'balls'
FILENAME = SIM + '-valid'
# FILENAME = SIM + '-train'
EPISODES_TRAIN = 1000
EPISODES_VALID = 200
EP_LEN = 100


class Record(object):
    def __init__(self):
        self.world = None
        self.ep = 0
        self.t = 0
        self.screen_list = []
        self.ep_list = []
        self.filename = os.path.join('data-balls/', FILENAME + '.pt')

    def run(self):
        total_episodes = EPISODES_TRAIN if 'train' in FILENAME else EPISODES_VALID
        while self.ep < total_episodes:
            print('Episode:', self.ep)
            # self.world = simple_sim.World()
            self.world = balls_sim.World()
            self.screen_list = []
            for _ in range(EP_LEN):
                self.world.run()
                observation = self.world.draw()
                self.screen_list.append(observation)
                self.t += 1

            self.ep += 1
            self.ep_list.append(self.screen_list[:])

    def write(self):
        data = np.array(self.ep_list)
        print('Writing', self.filename)
        torch.save(data, open(self.filename, 'wb'))


if __name__ == '__main__':
    rec = Record()

    rec.run()
    rec.write()



