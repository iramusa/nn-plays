#!/usr/bin/env python3

import numpy as np
import imageio

from structured_container import DataContainer


def batch_to_sequence(batch_eps, fpath, normalise=False):
    """

    :param batch_eps: in format (timesteps, batchs_size, im_channels, im_height, im_width)
    :param fpath:
    :return:
    """
    batch_eps = [batch_eps[:, i, ...] for i in range(batch_eps.shape[1])]
    batch_eps = np.concatenate(batch_eps, axis=-1)

    im_seq = []
    for i in range(batch_eps.shape[0]):
        im_seq.append(batch_eps[i, 0, :, :])

    imageio.mimsave(fpath, im_seq)


if __name__ == "__main__":

    data_test = DataContainer('data-balls/simple-test.pt', batch_size=16, ep_len_read=40)
    data_test.populate_images()
    batch_eps = data_test.get_batch_episodes()
    print(batch_eps.shape)

    batch_eps = batch_eps.transpose((1, 0, 4, 2, 3))

    batch_to_sequence(batch_eps, 'test.gif')

