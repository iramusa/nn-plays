import os
import numpy as np
import imageio

from torch_experiment import EP_LEN, UNCERTAIN_PERCEPTS, GUARANTEED_PERCEPTS

FOLDERS = ['images', 'network', 'numerical', 'plots']


def make_dir_tree(parent_dir):
    for folder in FOLDERS:
        new_dir = '{}/{}'.format(parent_dir, folder)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


def mask_percepts(images, p, return_indices=False):
    images_masked = np.copy(images)
    if p < 1.0:
        for_removal = np.random.random(EP_LEN) < p
    else:
        for_removal = np.ones(EP_LEN) > 0

    if UNCERTAIN_PERCEPTS > 0:
        clear_percepts = GUARANTEED_PERCEPTS + np.random.randint(0, UNCERTAIN_PERCEPTS)
    else:
        clear_percepts = GUARANTEED_PERCEPTS
    for_removal[0:clear_percepts] = False
    images_masked[:, for_removal, ...] = 0

    if return_indices:
        return images_masked, for_removal
    else:
        return images_masked


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

