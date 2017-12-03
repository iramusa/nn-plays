#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import os

from structured_container import DataContainer
from torch_nets import *
import my_utils


def test_autoencoder():
    BATCH_SIZE = 2
    EP_LEN = 20
    EPOCHS = 25
    UPDATES_PER_EPOCH = 100
    OUTPUT_FOLDER = "outputs_net_tests/"

    try:
        os.makedirs(OUTPUT_FOLDER)
    except OSError:
        pass

    data_test = DataContainer('data-balls/pass-train.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_test.populate_images()

    net = Autoencoder()
    criterion = nn.MSELoss()
    x = torch.FloatTensor(BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH)

    net = net.cuda()
    criterion = criterion.cuda()
    x = x.cuda()

    x = Variable(x)

    optimiser = optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(EPOCHS):
        for update in range(UPDATES_PER_EPOCH):
            batch = data_test.get_n_random_images(BATCH_SIZE)
            batch = batch.transpose((0, 3, 1, 2))
            batch = torch.FloatTensor(batch)

            net.zero_grad()

            x.data.resize_(batch.size()).copy_(batch)
            recon = net(x)
            err = criterion(recon, x)
            err.backward()

            optimiser.step()

            print('[%d/%d][%d/%d] Recon loss: %.4f'
                  % (epoch, EPOCHS, update, UPDATES_PER_EPOCH,
                     err.data[0]))

            if update % 100 == 0:
                vutils.save_image(recon.data,
                        '%s/reconstruction_epoch_%03d.png' % (OUTPUT_FOLDER, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(net.state_dict(), '%s/autencoder_epoch_%d.pth' % (OUTPUT_FOLDER, epoch))


def test_predictive_autoencoder():
    BATCH_SIZE = 4
    EP_LEN = 120
    V_SIZE = 1024
    GUARANTEED_PERCEPTS = 6
    UNCERTAIN_PERCEPTS = 4
    EPOCHS = 225
    UPDATES_PER_EPOCH = 1000
    OUTPUT_FOLDER = "outputs_net_tests/"

    try:
        os.makedirs(OUTPUT_FOLDER)
    except OSError:
        pass

    data_test = DataContainer('data-balls/2b-small-test.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_test.populate_images()
    # data_test.images.transpose((0, 1, 4, 2, 3))

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

    net = PredictiveAutoencoder(v_size=V_SIZE)
    criterion = nn.MSELoss()
    x = torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH)
    y = torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH)

    net = net.cuda()
    criterion = criterion.cuda()
    x = x.cuda()
    y = y.cuda()

    x = Variable(x)
    y = Variable(y)

    optimiser = optim.Adam(net.parameters(), lr=0.0002)

    net.load_state_dict(torch.load("outputs_net_tests/autencoder_epoch_3.pth"))

    for epoch in range(EPOCHS):
        for update in range(UPDATES_PER_EPOCH):
            batch = data_test.get_batch_episodes()
            # print(batch.shape)
            masked = mask_percepts(batch, p=0.9999)

            batch = batch.transpose((1, 0, 4, 2, 3))
            masked = masked.transpose((1, 0, 4, 2, 3))

            batch = torch.FloatTensor(batch)
            masked = torch.FloatTensor(masked)

            net.zero_grad()

            x.data.copy_(masked)
            y.data.copy_(batch)

            recon = net(x)
            err = criterion(recon, y)
            err.backward()

            optimiser.step()

            print('[%d/%d][%d/%d] Recon loss: %.4f'
                  % (epoch, EPOCHS, update, UPDATES_PER_EPOCH,
                     err.data[0]))

            if update % 100 == 0:
                recon_ims = recon.data.cpu().numpy()
                target_ims = y.data.cpu().numpy()
                joint = np.concatenate((target_ims, recon_ims), axis=-2)
                my_utils.batch_to_sequence(joint, fpath='%s/reconstruction_epoch_%03d.gif' % (OUTPUT_FOLDER, epoch))

            # if epoch == 0 and update == 0:
            #     my_utils.batch_to_sequence(y.data.cpu().numpy(), fpath='%s/orginal_epoch_%03d.gif' % (OUTPUT_FOLDER, epoch))

                # vutils.save_image(recon.data[:, 0, ...],
                #         '%s/reconstruction_epoch_%03d.png' % (OUTPUT_FOLDER, epoch),
                #         normalize=False)
                # vutils.save_image(x.data[:, 0, ...],
                #         '%s/original_epoch_%03d.png' % (OUTPUT_FOLDER, epoch),
                #         normalize=True)

        # do checkpointing
        torch.save(net.state_dict(), '%s/autencoder_epoch_%d.pth' % (OUTPUT_FOLDER, epoch))


if __name__ == "__main__":
    # test_autoencoder()
    test_predictive_autoencoder()
    pass
