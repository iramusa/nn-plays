#!/usr/bin/env python3

import argparse
import os

import torch_utils


PAE_BATCH_SIZE = 4
GAN_BATCH_SIZE = 16
AVERAGING_BATCH_SIZE = 16
# EP_LEN = 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train network based on time-series data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', default='default_experiment', type=str,
                        help="Folder for the outputs.")
    parser.add_argument('--dataset_type', type=str,
                        choices=['balls', 'atari', 'stocks'],
                        help="Type of the dataset.")
    parser.add_argument('--data_folder', type=str,
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
                        choices=['pae', 'paegan', 'paegan-sampler'],
                        help="Different training modes enable training of different parts of the network")
    parser.add_argument('--compare_with_pf', default=False, type=bool,
                        help="Should the results be compared with particle filter?")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        torch_utils.make_dir_tree(args.output_dir)

    if args.dataset_type == 'balls':
        sim_config = None
        train_getter = None
        valid_getter = None
    else:
        raise ValueError('Wrong dataset type {}'.format(args.dataset_type))


