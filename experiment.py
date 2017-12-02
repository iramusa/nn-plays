#!/usr/bin/env python3

import os
import datetime

import numpy as np
import copy

from structured_recorder import Record
from structured_container import DataContainer
from balls_sim import DEFAULT_SIM_CONFIG
from hydranet import HydraNet, EP_LEN, BATCH_SIZE

import pandas as pd  # leads to error in combo with hydranet if imported earlier
import matplotlib.pyplot as plt # leads to error in combo with hydranet if imported earlier

FOLDER_EXPS = 'experiments'

train_scheme = {
    'clear_batches': 0,  # how many batches?
    'lr_initial': 0.0005,
    'guaranteed_percepts': 5,  # how many first percepts are guaranteed to be non-masked?
    'uncertain_percepts': 8,  # how many further have a high chance to be non-masked?
    'p_levels': np.sqrt(np.linspace(0.05, 0.99 ** 2, 10)).tolist(),  # progressing probabilities of masking percepts
    'p_level_batches': 1500,  # how many batches per level
    'p_final': 0.99,  # final probability level
    'lr_final': 0.0002,
    'final_batches': 2000,  # number of batches for final training
    'max_until_convergence': 1600,
    'v_size': 128,  #
}

sim_config = {
    'n_bodies': 1,
    'radius_mode': 'uniform',
    'radius': 2.0,
    'mass_mode': 'uniform',
    'mass': 1.0,
    'wall_action': 'pass',
    'ball_action': 'pass',
    'measurement_noise': 0.001,
    'dynamics_noise': 0.004,
}

train_config = {
    'sim_type': 'diff',
    'sim_config': sim_config,
    'train': 'train',
    'n_episodes': 1000,
    'episode_length': EP_LEN,
    'folder': 'data-balls/',
    'random_seed': 0
}

valid_config = copy.deepcopy(train_config)
valid_config['sim_config'] = sim_config
valid_config['n_episodes'] = 200
valid_config['train'] = 'valid'
valid_config['random_seed'] += 1

GIFS_NO = 10


class Experiment(object):
    def __init__(self, ctrl_var, var_vals, exp_name):
        self.date = datetime.datetime.now().strftime('%y-%m-%d_%H:%M')
        self.sim_conf = sim_config
        self.train_scheme = train_scheme

        self.train_config = train_config
        self.valid_config = valid_config

        self.ctrl_var = ctrl_var
        self.var_vals = var_vals
        self.exp_name = exp_name

        # folders
        self.folder_top = '{}/{}-{}/'.format(FOLDER_EXPS, self.date,  self.exp_name)
        self.folder_data = '{}/{}-{}/data/'.format(FOLDER_EXPS, self.date,  self.exp_name)
        self.folder_gifs = '{}/{}-{}/gifs/'.format(FOLDER_EXPS, self.date, self.exp_name)
        self.folder_modules = '{}/{}-{}/modules/'.format(FOLDER_EXPS, self.date, self.exp_name)
        self.folder_numerical = '{}/{}-{}/nums/'.format(FOLDER_EXPS, self.date, self.exp_name)
        self.folder_plots = '{}/{}-{}/plots/'.format(FOLDER_EXPS, self.date, self.exp_name)

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

        # print('Loading base models')
        # tag = 'base-{}'.format(v_size)
        # self.net.load_modules(self.folder_base_models, tag=tag)

        print('Starting training')

        self.net.execute_scheme(self.train_box.get_batch_episodes, self.valid_box.get_batch_episodes)
        self.net.save_modules(self.folder_modules, tag='{}'.format(v_size))

        print('Saving base models')
        tag = 'base-{}'.format(v_size)
        self.net.save_modules(self.folder_base_models, tag=tag)

        print('Recording videos')
        pae_losseses = []
        pf_losseses = []
        for j in range(GIFS_NO):
            losses = self.net.draw_pred_gif(self.valid_box.get_n_random_episodes_full, p=1.0, use_stepper=False, use_pf=True,
                                   sim_config=sim_config, folder_plots=self.folder_gifs, tag='{}-{}'.format(val, j),
                                   normalize=True)
            pae_losseses.append(losses['pae_losses'])
            pf_losseses.append(losses['pf_losses'])

        baseline_level = self.net.plot_losses(folder_plots=self.folder_plots, tag=val,
                                              image_getter=self.valid_box.get_batch_episodes)

        av_pae_losses = np.mean(np.array(pae_losseses), axis=0)
        av_pf_losses = np.mean(np.array(pf_losseses), axis=0)
        self.write_av_losses(av_pae_losses, av_pf_losses, baseline_level, val)


        # get numericals
        self.x.append(val)
        self.train_errors.append(self.get_errors(self.train_box.get_batch_episodes))
        self.valid_errors.append(self.get_errors(self.valid_box.get_batch_episodes))

    def write_av_losses(self, pae_loss, pf_loss, baseline_level, tag):
        plt.clf()

        plt.plot(pae_loss)
        plt.plot(pf_loss)
        baseline = np.ones(len(pae_loss)) * baseline_level
        plt.plot(baseline, 'g--')

        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('timestep')
        plt.legend(['PAE', 'PF', 'baseline'])
        fpath = '{}/av_time_losses-{}.png'.format(self.folder_plots, tag)
        plt.savefig(fpath)

    def write_losses(self):
        plt.clf()

        results = pd.DataFrame({
            self.ctrl_var: self.x,
            'train_error': self.train_errors,
            'valid_error': self.valid_errors
        })
        fpath = '{}/errors.csv'.format(self.folder_numerical)
        results.to_csv(fpath)

        plt.scatter(self.x, self.train_errors)
        plt.scatter(self.x, self.valid_errors)

        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel(self.ctrl_var)
        plt.legend(['train', 'valid'])
        fpath = '{}/run-summary.png'.format(self.folder_plots)
        plt.savefig(fpath)

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
        self.train_box = DataContainer(fpath_train, batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
        self.train_box.populate_images()

        rec = Record(**self.valid_config)
        rec.run()
        fpath_valid = '{}/test.pt'.format(self.folder_data)
        rec.write(fpath_valid)
        self.valid_box = DataContainer(fpath_valid, batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
        self.valid_box.populate_images()

    def generate_report(self):
        html_doc = "<html><head><title>{0}</title></head>\n<body>\n".format(self.exp_name)
        html_doc += "<h2>{0}</h2>\n".format(self.exp_name)
        html_doc += "Simulation configuration: {0}\n".format(self.sim_conf)
        html_doc += "Controlled variable: {0}\n".format(self.ctrl_var)
        html_doc += "Range: {0}\n".format(self.var_vals)

        html_doc += "<h3>Loss for different values of {}</h3>\n".format(self.ctrl_var)
        html_doc += "<center><img src=\"plots/run-summary.png\" width=\"800\"></center>".format(self.folder_plots)

        for val in self.var_vals:
            html_doc += "<h3>{0} set to {1}</h3>\n".format(self.ctrl_var, val)
            html_doc += "<center><img src=\"plots/loss-{}.png\" width=\"800\"></center>\n".format(val)

            html_doc += "<h3>Loss over time for {} set to {}</h3>\n".format(self.ctrl_var, val)
            html_doc += "<center><img src=\"plots/av_time_losses-{}.png\" width=\"800\"></center>\n".format(val)

            table = "<center><table style=\"text-align: center;\" style=\"margin: 0px auto;\" border=\"1\">" \
                    "<tr>" \
                    "<td>{0}</td>" \
                    "<td>percept</td>" \
                    "<td>ground truth</td>" \
                    "<td>prediction</td>" \
                    "<td>particle filter</td>" \
                    "</tr>\n".format(self.ctrl_var)

            for i in range(GIFS_NO):
                new_row =   "<tr>" \
                                "<td>{2}</td>" \
                                "<td><img src=\"gifs/percepts-{2}-{3}.gif\" width=\"140\"></td>" \
                                "<td><img src=\"gifs/truths-{2}-{3}.gif\" width=\"140\"></td>" \
                                "<td><img src=\"gifs/pae_preds-{2}-{3}.gif\" width=\"140\"></td>" \
                                "<td><img src=\"gifs/pf_preds-{2}-{3}.gif\" width=\"140\"></td>" \
                            "</tr>\n".format(self.ctrl_var, self.folder_gifs, val, i)
                table += new_row

            table += "</table></center>\n"
            html_doc += table

        html_doc += "</body>"

        rep = open('{}/report.html'.format(self.folder_top), 'w')
        rep.write(html_doc)


if __name__ == '__main__':
    exp_name = 'wp_1b_d004_small'
    ctrl_var = 'v_size'
    var_vals = [128, 32, 64, 256, 512, 1024]
    # var_vals = [256, 64, 128, 512]

    exp = Experiment(ctrl_var, var_vals, exp_name)
    exp.generate_report()
    exp.run()
    exp.write_losses()














