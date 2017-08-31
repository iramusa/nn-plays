#!/usr/bin/env python3

import os
import datetime

import numpy as np

from structured_recorder import Record
from structured_container import DataContainer
from balls_sim import DEFAULT_SIM_CONFIG
from hydranet import HydraNet

import pandas as pd  # leads to error in combo with hydranet if imported earlier

FOLDER_EXPS = 'experiments'

BATCH_SIZE = 32
SERIES_SHIFT = 1
EP_LEN = 100 - SERIES_SHIFT


train_scheme = {
    'clear_training': True,  # run training on clear episodes (non-masked percepts)
    # 'clear_training': False,  # run training on clear episodes (non-masked percepts)
    'clear_batches': 500,  # how many batches?
    'lr_initial': 0.001,
    'guaranteed_percepts': 5,  # how many first percepts are guaranteed to be non-masked?
    'uncertain_percepts': 8,  # how many further have a high chance to be non-masked?
    'p_levels': np.sqrt(np.linspace(0.05, 0.99, 10)).tolist(),  # progressing probabilities of masking percepts
    # 'p_levels': [],  # progressing probabilities of masking percepts
    'p_level_batches': 400,  # how many batches per level
    'p_final': 0.99,  # final probability level
    'lr_final': 0.0002,
    'final_batches': 1500,  # number of batches for final training
    # 'v_size': 64, # sufficient for near no noise
    'v_size': 128,  #
}

sim_config = {
    'n_bodies': 1,
    'radius_mode': 'uniform',
    'radius': 3.5,
    'mass_mode': 'uniform',
    'mass': 1.0,
    'wall_action': 'pass',
    'ball_action': 'pass',
    'measurement_noise': 0.0,
    'dynamics_noise': 0.000000001,
}

train_config = {
    'sim_type': 'easy',
    'sim_config': sim_config,
    'train': 'train',
    'n_episodes': 1000,
    'episode_length': 100,
    'folder': 'data-balls/',
    'random_seed': 0
}

valid_config = {
    'sim_type': 'easy',
    'sim_config': sim_config,
    'train': 'valid',
    'n_episodes': 500,
    'episode_length': 100,
    'folder': 'data-balls/',
    'random_seed': 0
}


class Experiment(object):
    def __init__(self, ctrl_var, var_vals, exp_name):
        self.date = datetime.datetime.now().strftime('%y-%m-%d_%H:%M')
        self.sim_conf = DEFAULT_SIM_CONFIG
        self.train_scheme = train_scheme

        self.train_config = train_config
        self.valid_config = valid_config

        self.ctrl_var = ctrl_var
        self.var_vals = var_vals
        self.exp_name = exp_name

        # folders
        self.folder_data = '{}/{}-{}/data/'.format(FOLDER_EXPS, self.exp_name, self.date)
        self.folder_gifs = '{}/{}-{}/gifs/'.format(FOLDER_EXPS, self.exp_name, self.date)
        self.folder_modules = '{}/{}-{}/modules/'.format(FOLDER_EXPS, self.exp_name, self.date)
        self.folder_numerical = '{}/{}-{}/nums/'.format(FOLDER_EXPS, self.exp_name, self.date)
        self.folder_plots = '{}/{}-{}/plots/'.format(FOLDER_EXPS, self.exp_name, self.date)

        self.folder_base_models = 'base_models/'

        self.folders = [self.folder_modules, self.folder_data, self.folder_base_models, self.folder_gifs,
                        self.folder_numerical, self.folder_plots]
        self.make_folders()

        self.train_box = None
        self.valid_box = None
        self.net = None

        self.x = []
        self.train_errors = []
        self.valid_errors = []

    def make_folders(self):
        for folder in self.folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def run(self):
        for i, val in enumerate(self.var_vals):
            self.run_single(val, i)

        results = pd.DataFrame({
            self.ctrl_var: self.x,
            'train_error': self.train_errors,
            'valid_error': self.valid_errors
        })
        fpath = '{}/errors.csv'.format(self.folder_numerical)
        results.to_csv(fpath)

    def run_single(self, val, i):
        print('Setting {} to {}'.format(ctrl_var, val))

        if self.ctrl_var in self.sim_conf.keys():
            self.sim_conf[self.ctrl_var] = val
            self.train_config['sim_config'] = self.sim_conf
            self.valid_config['sim_config'] = self.sim_conf
            self.generate_data()

        elif self.ctrl_var == 'v_size':
            self.train_scheme[self.ctrl_var] = val
            if self.train_box is None:
                self.generate_data()
        else:
            raise ValueError('Bad ctrl_var {}'.format(self.ctrl_var))

        v_size = self.train_scheme['v_size']
        self.net = HydraNet(**self.train_scheme)
        tag = 'base-{}'.format(v_size)
        self.net.load_modules(self.folder_base_models, tag=tag)

        print('Starting training')

        self.net.execute_scheme(self.train_box.get_batch_episodes, self.valid_box.get_batch_episodes)
        self.net.save_modules(self.folder_modules, tag='base-{}'.format(v_size))
        self.net.draw_pred_gif(self.valid_box.get_n_random_episodes_full, p=1.0, use_stepper=False, use_pf=False,
                               folder_plots=self.folder_gifs, tag=val)
        self.net.plot_losses(folder_plots=self.folder_plots, tag=val)

        # get numericals
        self.x.append(val)
        self.train_errors.append(self.get_errors(self.train_box.get_batch_episodes))
        self.valid_errors.append(self.get_errors(self.valid_box.get_batch_episodes))

    def get_errors(self, data_getter, test_iters=20):
        error_cum = 0
        for j in range(test_iters):
            error_cum += self.net.train_batch_pred_ae(data_getter, p=1.0, test=True)

        error = error_cum / test_iters
        return error

    def generate_data(self):
        print('Generating data')

        rec = Record(**self.train_config)
        rec.run()
        fpath_train = '{}/train.pt'.format(self.folder_data)
        rec.write(fpath_train)
        self.train_box = DataContainer(fpath_train, batch_size=32, ep_len_read=EP_LEN)
        self.train_box.populate_images()

        rec = Record(**self.valid_config)
        rec.run()
        fpath_valid = '{}/test.pt'.format(self.folder_data)
        rec.write(fpath_valid)
        self.valid_box = DataContainer(fpath_valid, batch_size=32, ep_len_read=EP_LEN)
        self.valid_box.populate_images()


if __name__ == '__main__':
    exp_name = 'bases'
    ctrl_var = 'v_size'
    var_vals = [16, 32, 64, 128, 256, 512]

    exp = Experiment(ctrl_var, var_vals, exp_name)
    exp.run()













