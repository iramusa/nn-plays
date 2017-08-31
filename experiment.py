#!/usr/bin/env python3

import datetime
import numpy as np

from structured_recorder import Record
from structured_container import DataContainer
from balls_sim import DEFAULT_SIM_CONFIG
from hydranet import HydraNet

FOLDER_EXPS = 'experiments'
# FOLDER_DIAGRAMS = 'diagrams'
# FOLDER_MODELS = 'models'
# FOLDER_PLOTS = 'plots'


BATCH_SIZE = 32
SERIES_SHIFT = 1
EP_LEN = 100 - SERIES_SHIFT


DEFAULT_TRAIN_SCHEME = {
    'clear_training': True,  # run training on clear episodes (non-masked percepts)
    # 'clear_training': False,  # run training on clear episodes (non-masked percepts)
    'clear_batches': 1,  # how many batches?
    'lr_initial': 0.001,
    'guaranteed_percepts': 5,  # how many first percepts are guaranteed to be non-masked?
    'uncertain_percepts': 8,  # how many further have a high chance to be non-masked?
    'p_levels': np.sqrt(np.linspace(0.05, 0.99, 10)).tolist(),  # progressing probabilities of masking percepts
    # 'p_levels': [],  # progressing probabilities of masking percepts
    'p_level_batches': 1,  # how many batches per level
    'p_final': 0.99,  # final probability level
    'lr_final': 0.0002,
    'final_batches': 1,  # number of batches for final training
    # 'v_size': 64, # sufficient for near no noise
    'v_size': 128,  #
}

train_config = {
    'sim_type': 'bounce',
    'sim_config': DEFAULT_SIM_CONFIG,
    'train': 'train',
    'n_episodes': 1000,
    'episode_length': 100,
    'folder': 'data-balls/',
    'random_seed': 0
}

valid_config = {
    'sim_type': 'bounce',
    'sim_config': DEFAULT_SIM_CONFIG,
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
        self.train_scheme = DEFAULT_TRAIN_SCHEME

        self.train_config = train_config
        self.valid_config = valid_config

        self.ctrl_var = ctrl_var
        self.var_vals = var_vals
        self.exp_name = exp_name

        # folders
        self.folder_data = '{}/{}/data/'.format(FOLDER_EXPS, exp_name)
        self.folder_gifs = '{}/{}/gifs/'.format(FOLDER_EXPS, exp_name)
        self.folder_models = '{}/{}/models/'.format(FOLDER_EXPS, exp_name)
        self.folder_numerical = '{}/{}/nums/'.format(FOLDER_EXPS, exp_name)
        self.folder_plots = '{}/{}/plots/'.format(FOLDER_EXPS, exp_name)

        self.folder_base_models = 'base_models/'

        self.train_box = None
        self.valid_box = None
        self.net = None

        self.x = []
        self.train_error = []
        self.valid_error = []

    def run(self):
        for i, val in enumerate(self.var_vals):
            self.run_single(val, i)

    def run_single(self, val, i):
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
        tag = 'base-{}'.format(self.folder_models, v_size)
        self.net.load_modules(self.folder_models, tag=tag)

        self.net.execute_scheme(self.train_box.get_batch_episodes, self.valid_box.get_batch_episodes)

    def generate_data(self):
        rec = Record(**self.train_config)
        rec.run()
        fpath_train = '{}/train.pt'.format(self.folder_data)
        rec.write(fpath_train)
        self.train_box = DataContainer(fpath_train, batch_size=32, ep_len_read=EP_LEN)
        self.train_box.populate_images()

        rec = Record(**self.valid_config)
        rec.run()
        fpath_valid = '{}/train.pt'.format(self.folder_data)
        rec.write(fpath_valid)
        self.valid_box = DataContainer(fpath_valid, batch_size=32, ep_len_read=EP_LEN)
        self.valid_box.populate_images()


if __name__ == '__main__':
    exp = Experiment()
    exp.run()














