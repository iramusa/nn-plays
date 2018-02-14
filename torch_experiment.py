#!/usr/bin/env python3

import argparse
import os
import torch

import torch_utils
from structured_container import DataContainer
from torch_nets import PAEGAN

PAE_BATCH_SIZE = 4
GAN_BATCH_SIZE = 16
AVERAGING_BATCH_SIZE = 16
EP_LEN = 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train network based on time-series data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', default='default_experiment', type=str,
                        help="Folder for the outputs.")
    parser.add_argument('--dataset_type', default='balls', type=str,
                        choices=['balls', 'atari', 'stocks'],
                        help="Type of the dataset.")
    parser.add_argument('--data_dir', type=str,
                        help="Folder with the data")
    parser.add_argument('--start_from_checkpoint', type=int,
                        help="Use network that was trained already")
    parser.add_argument('--epochs', default=10, type=int,
                        help="How many epochs to train for?")
    parser.add_argument('--updates_per_epoch', default=1000, type=int,
                        help="How many updates per epoch?")
    parser.add_argument('--training_stage', type=str,
                        choices=['pae', 'paegan', 'paegan-sampler'],
                        help="Different training modes enable training of different parts of the network")
    parser.add_argument('--mask_probability', default=0.99, type=float,
                        help="What fraction of input observations is masked?")
    parser.add_argument('--compare_with_pf', default=False, type=bool,
                        help="Should the results be compared with particle filter?")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        torch_utils.make_dir_tree(args.output_dir)

    sim_config = None
    if args.dataset_type == 'balls':
        sim_config = torch.load('{}/train.conf'.format(args.data_dir))
        train_container = DataContainer('{}/train.pt'.format(args.data_dir), batch_size=PAE_BATCH_SIZE)
        valid_container = DataContainer('{}/test.pt'.format(args.data_dir), batch_size=PAE_BATCH_SIZE)
        train_container.populate_images()
        valid_container.populate_images()

        train_getter = train_container.get_batch_episodes
        valid_getter = valid_container.get_batch_episodes

        net = PAEGAN()

    else:
        raise ValueError('Wrong dataset type {}'.format(args.dataset_type))


