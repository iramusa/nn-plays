#!/usr/bin/env python3
"""
Records structured data from bouncing balls simulation
"""
import balls_sim
import os
import numpy as np
import torch

simulation_config = {
    'n_bodies': 1,
    'radius_mode': 'uniform',
    'radius': 3.5,
    'mass_mode': 'uniform',
    'mass': 1.0,
    'wall_action': 'pass',
    'ball_action': 'pass',
    'measurement_noise': 0.0,
}

record_config = {
    'sim_type': 'balls',
    'sim_config': simulation_config,
    'train': 'train',
    # 'train': 'valid',
    # 'n_episodes': 500,
    'n_episodes': 1000,
    'episode_length': 100,
    'folder': 'data-balls/',
    'random_seed': 0
}


class Record(object):
    def __init__(self, **kwargs):
        self.record = {}
        self.record.update(kwargs)

        self.filename = '{0}-{1}.pt'.format(kwargs['sim_type'],
                                            kwargs['train'])
        self.filepath = os.path.join(kwargs['folder'], self.filename)

        self.n_episodes = kwargs['n_episodes']
        self.ep_length = kwargs['episode_length']
        self.all_eps = []

        self.record.update({'episodes': self.all_eps})

        np.random.seed(kwargs['random_seed'])

    def run(self):
        sim_config = self.record['sim_config']

        for i_ep in range(self.n_episodes):
            print(i_ep)
            ep_dict = {}
            t_list = []
            ep_dict.update({'t_list': t_list})

            world = balls_sim.World(**sim_config)

            # save constants for the episode
            radii = [np.copy(body.r) for body in world.bodies]
            masses = [np.copy(body.m) for body in world.bodies]
            ep_dict.update({'radii': radii})
            ep_dict.update({'masses': masses})

            for t in range(self.ep_length):
                world.run()
                t_dict = {}
                poses = [np.copy(body.pos) for body in world.bodies]
                # vels = [body.vel for body in world.bodies]
                measures = [np.copy(body.measured_pos) for body in world.bodies]

                t_dict.update({'poses': poses})
                t_dict.update({'measures': measures})
                t_list.append(t_dict)

            self.all_eps.append(ep_dict)

    def write(self):
        # data = np.array(self.ep_list)
        print('Writing', self.filepath)
        torch.save(self.record, open(self.filepath, 'wb'))


if __name__ == '__main__':
    rec = Record(**record_config)

    rec.run()
    rec.write()



