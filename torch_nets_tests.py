#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import os

from structured_container import DataContainer
from torch_nets import *

if __name__ == "__main__":
    BATCH_SIZE = 32
    EP_LEN = 100
    EPOCHS = 25
    UPDATES_PER_EPOCH = 100
    OUTPUT_FOLDER = "outputs_net_tests/"

    try:
        os.makedirs(OUTPUT_FOLDER)
    except OSError:
        pass

    data_test = DataContainer('data-balls/bounce-valid.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
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
