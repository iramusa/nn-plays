#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
import os
from tqdm import tqdm, trange

from structured_container import DataContainer
from torch_nets import *
import my_utils

BATCH_SIZE = 4
GAN_BATCH_SIZE = 16
AVERAGING_BATCH_SIZE = 24
EP_LEN = 120
GUARANTEED_PERCEPTS = 6
UNCERTAIN_PERCEPTS = 4
P_NO_OBS = 0.75
EPOCHS = 225
UPDATES_PER_EPOCH = 10000
EXP_FOLDER = "/home/ira/code/projects/nn-play/experiments/paegan-state/"
DATA_FOLDER = "{}/data".format(EXP_FOLDER)


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


def test_autoencoder():
    try:
        os.makedirs(EXP_FOLDER)
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
                        '%s/reconstruction_epoch_%03d.png' % (EXP_FOLDER, epoch),
                                  normalize=True)

        # do checkpointing
        torch.save(net.state_dict(), '%s/autencoder_epoch_%d.pth' % (EXP_FOLDER, epoch))


def train_predictive_autoencoder():
    try:
        os.makedirs(EXP_FOLDER)
    except OSError:
        pass

    data_train = DataContainer('/home/ira/code/projects/nn-play/experiments/0__well_done/17-11-30_09:05-wp_1b_1l_small_deter/data/train.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_test = DataContainer('/home/ira/code/projects/nn-play/experiments/0__well_done/17-11-30_09:05-wp_1b_1l_small_deter/data/test.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_train.populate_images()
    data_test.populate_images()

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

    start_at_epoch = 5
    net.load_state_dict(torch.load("%s/autencoder_epoch_%d.pth" % (EXP_FOLDER, start_at_epoch - 1)))

    postfix = {}
    for epoch in range(start_at_epoch, EPOCHS):
        bar = trange(UPDATES_PER_EPOCH)
        postfix['epoch'] = '[%d/%d]' % (epoch, EPOCHS)
        for update in bar:

            batch = data_train.get_batch_episodes()
            masked = mask_percepts(batch, p=0.98)

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

            postfix['train loss'] = err.data[0]

            if update % 500 == 0:
                recon_ims = recon.data.cpu().numpy()
                target_ims = y.data.cpu().numpy()
                joint = np.concatenate((target_ims, recon_ims), axis=-2)
                my_utils.batch_to_sequence(joint, fpath='%s/training_recon_%03d.gif' % (EXP_FOLDER, epoch))

            if update % 10 == 0:
                batch = data_test.get_batch_episodes()
                masked = mask_percepts(batch, p=1.0)

                masked = masked.transpose((1, 0, 4, 2, 3))
                masked = torch.FloatTensor(masked)
                x.data.copy_(masked)

                batch = batch.transpose((1, 0, 4, 2, 3))
                batch = torch.FloatTensor(batch)
                y.data.copy_(batch)

                recon = net(x)
                err = criterion(recon, y)
                postfix['valid loss'] = err.data[0]

            if update % 500 == 0:
                recon_ims = recon.data.cpu().numpy()
                target_ims = y.data.cpu().numpy()
                joint = np.concatenate((target_ims, recon_ims), axis=-2)
                my_utils.batch_to_sequence(joint, fpath='%s/valid_recon_%03d.gif' % (EXP_FOLDER, epoch))

            bar.set_postfix(**postfix)

        # do checkpointing
        torch.save(net.state_dict(), '%s/autencoder_epoch_%d.pth' % (EXP_FOLDER, epoch))


def train_PAEGAN(start_at_epoch=0, train_gan=True, train_av=True, n_epochs=EPOCHS):
    try:
        os.makedirs(EXP_FOLDER)
    except OSError:
        pass

    data_train = DataContainer('/home/ira/code/projects/nn-play/experiments/0__well_done/17-11-30_09:05-wp_1b_1l_small_deter/data/train.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_test = DataContainer('/home/ira/code/projects/nn-play/experiments/0__well_done/17-11-30_09:05-wp_1b_1l_small_deter/data/test.pt', batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_train.populate_images()
    data_test.populate_images()

    if torch.cuda.is_available():
        net = VisualPAEGAN(v_size=V_SIZE, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE).cuda()
        criterion_pae = nn.MSELoss().cuda()
        criterion_gan = nn.BCELoss().cuda()
        # criterion_gan = nn.MSELoss().cuda()
        criterion_gen_averaged = nn.MSELoss().cuda()

        obs_in = Variable(torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH).cuda())
        obs_out = Variable(torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH).cuda())

        averaging_noise = Variable(torch.FloatTensor(AVERAGING_BATCH_SIZE, N_SIZE).cuda())
        noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE).cuda())

        fixed_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE).normal_(0, 1).cuda())
        fixed_bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, BS_SIZE).uniform_(-1, 1).cuda())
        label = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1).cuda())
    else:
        net = VisualPAEGAN(v_size=V_SIZE, bs_size=BS_SIZE, n_size=N_SIZE, g_size=G_SIZE)
        criterion_pae = nn.MSELoss()
        criterion_gan = nn.BCELoss()
        # criterion_gan = nn.MSELoss()
        criterion_gen_averaged = nn.MSELoss()

        obs_in = Variable(torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))
        obs_out = Variable(torch.FloatTensor(EP_LEN, BATCH_SIZE, IM_CHANNELS, IM_WIDTH, IM_WIDTH))

        averaging_noise = Variable(torch.FloatTensor(AVERAGING_BATCH_SIZE, N_SIZE))
        noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE))

        fixed_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE).normal_(0, 1))
        fixed_bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, BS_SIZE).uniform_(-1, 1))
        label = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1))

    real_label = 1
    fake_label = 0

    optimiser_pae = optim.Adam([{'params': net.bs_prop.parameters()},
                                {'params': net.decoder.parameters()}],
                               lr=0.0004)
    # optimiser_g = optim.Adam([{'params': net.bs_prop.parameters(), 'lr': 0.0001},
    #                           {'params': net.G.parameters(), 'lr': 0.0002}])
    optimiser_g = optim.Adam(net.G.parameters(), lr=0.0002)
    optimiser_sum = optim.Adam(net.G.parameters(), lr=0.0002)
    optimiser_d = optim.Adam(net.D.parameters(), lr=0.0002)

    if start_at_epoch > 0:
        net.load_state_dict(torch.load("%s/paegan_epoch_%d.pth" % (EXP_FOLDER, start_at_epoch - 1)))

    postfix = {}
    for epoch in range(start_at_epoch, n_epochs):
        bar = trange(UPDATES_PER_EPOCH)
        postfix['epoch'] = '[%d/%d]' % (epoch, EPOCHS)
        for update in bar:
            batch = data_train.get_batch_episodes()
            masked = mask_percepts(batch, p=P_NO_OBS)

            batch = batch.transpose((1, 0, 4, 2, 3))
            masked = masked.transpose((1, 0, 4, 2, 3))

            batch = torch.FloatTensor(batch)
            masked = torch.FloatTensor(masked)

            net.zero_grad()

            obs_in.data.copy_(masked)
            obs_out.data.copy_(batch)

            # generate beliefs states
            states = net.bs_prop(obs_in)
            states = states.view(EP_LEN * BATCH_SIZE, -1)

            # generate observation expectations, compute pae gradients, update
            obs_expectation = net.decoder(states)
            obs_expectation = obs_expectation.view(obs_in.size())

            err_pae = criterion_pae(obs_expectation, obs_out)
            err_pae.backward(retain_graph=True)
            optimiser_pae.step()
            postfix['pae train loss'] = err_pae.data[0]

            if train_gan is True:
                net.zero_grad()

                # # flip labels
                # if update % 2000 == 0:
                #     if real_label == 1:
                #         real_label = 0
                #         fake_label = 1
                #     else:
                #         real_label = 1
                #         fake_label = 0
                # train discriminator with real data
                label.data.fill_(real_label)
                obs_out_non_ep = obs_out.view(EP_LEN * BATCH_SIZE,
                                              obs_out.size(2), obs_out.size(3), obs_out.size(4))

                obs_out_non_ep = obs_out_non_ep.unfold(0, 1, (EP_LEN*BATCH_SIZE)//GAN_BATCH_SIZE).squeeze(-1)
                states_non_ep = states.unfold(0, 1, (EP_LEN*BATCH_SIZE)//GAN_BATCH_SIZE).squeeze(-1)

                out_D_real = net.D(obs_out_non_ep)
                err_D_real = criterion_gan(out_D_real, label)
                err_D_real.backward()
                # D_x = out_D_real.data.mean()

                # print('out_D_real', out_D_real.data)
                # print('err_D_real', err_D_real.data[0])

                # train discriminator with fake data
                noise.data.normal_(0, 1)
                obs_sample = net.G(noise, states_non_ep)
                obs_sample = net.decoder(obs_sample)
                label.data.fill_(fake_label)
                out_D_fake = net.D(obs_sample.detach())
                err_D_fake = criterion_gan(out_D_fake, label)
                err_D_fake.backward()
                # D_G_z1 = out_D_fake.data.mean()

                # print('out_D_fake', out_D_fake.data)
                # print('err_D_fake', err_D_fake.data[0])

                err_D = (err_D_fake + err_D_real)/2
                optimiser_d.step()

                # train generator using discriminator
                net.zero_grad()

                batch = data_train.get_batch_episodes()
                masked = mask_percepts(batch, p=P_NO_OBS)

                batch = batch.transpose((1, 0, 4, 2, 3))
                masked = masked.transpose((1, 0, 4, 2, 3))

                batch = torch.FloatTensor(batch)
                masked = torch.FloatTensor(masked)

                obs_in.data.copy_(masked)
                obs_out.data.copy_(batch)

                # generate beliefs states
                states = net.bs_prop(obs_in)
                states = states.view(EP_LEN * BATCH_SIZE, -1)
                states_non_ep = states.unfold(0, 1, (EP_LEN*BATCH_SIZE)//GAN_BATCH_SIZE).squeeze(-1)

                noise.data.normal_(0, 1)
                obs_sample = net.G(noise, states_non_ep)
                obs_sample = net.decoder(obs_sample)

                label.data.fill_(real_label)
                out_D_fake = net.D(obs_sample)
                # print('out d fake', out_D_fake)
                err_G = criterion_gan(out_D_fake, label)
                err_G.backward()
                # D_G_z2 = out_D_fake.data.mean()
                optimiser_g.step()
                postfix['g train loss'] = err_G.data[0]
                postfix['d train loss'] = err_D.data[0]

                if update % 500 == 0:
                    vutils.save_image(obs_out_non_ep.data,
                            '%s/real_samples.png' % EXP_FOLDER,
                            normalize=True)
                    obs_sample = net.G(fixed_noise, fixed_bs_noise)
                    obs_sample = net.decoder(obs_sample)
                    vutils.save_image(obs_sample.data,
                            '%s/fake_samples_epoch_%03d.png' % (EXP_FOLDER, epoch),
                            normalize=False)
                    obs_recon = net.decoder(fixed_bs_noise)
                    vutils.save_image(obs_recon.data,
                            '%s/expectation_samples_epoch_%03d.png' % (EXP_FOLDER, epoch),
                            normalize=False)

            if train_av is True:
                net.zero_grad()
                # train generator using averaging
                # pull a 50th state from 0th episode in the last batch
                state = states[50:51, ...]
                # print('state size', state.size())
                state = state.expand(AVERAGING_BATCH_SIZE, -1)
                # print('state size', state.size())
                # get corresponding observation expectation
                obs_exp = obs_expectation[50:51, 0, ...]
                # print('obs size', obs_exp.size())

                # generate samples from state
                averaging_noise.data.normal_(0, 1)
                n_samples = net.G(averaging_noise, state.detach())
                # print('samples size', n_samples.size())

                sample_av = n_samples.mean(dim=0)
                sample_av = sample_av.unsqueeze(0)
                # print('samples av size', sample_av.size())
                # print('obs_exp size', obs_exp.size())

                err_sum = criterion_gen_averaged(sample_av, obs_exp.detach())
                err_sum.backward()
                optimiser_sum.step()
                postfix['sum train loss'] = err_sum.data[0]

                if update % 100 == 0:
                    sample_mixture = sample_av.data.cpu().numpy()
                    observation_belief = obs_exp.data.cpu().numpy()
                    joint = np.concatenate((observation_belief, sample_mixture), axis=-2)
                    joint = np.expand_dims(joint, axis=0)
                    my_utils.batch_to_sequence(joint, fpath='%s/training_sum_%03d.gif' % (EXP_FOLDER, epoch))

            if update % 500 == 0:
                recon_ims = obs_expectation.data.cpu().numpy()
                target_ims = obs_out.data.cpu().numpy()
                joint = np.concatenate((target_ims, recon_ims), axis=-2)
                my_utils.batch_to_sequence(joint, fpath='%s/training_recon_%03d.gif' % (EXP_FOLDER, epoch))

            if update % 100 == 0:
                batch = data_test.get_batch_episodes()
                masked = mask_percepts(batch, p=P_NO_OBS)
                # masked = mask_percepts(batch, p=1)

                masked = masked.transpose((1, 0, 4, 2, 3))
                masked = torch.FloatTensor(masked)
                obs_in.data.copy_(masked)

                batch = batch.transpose((1, 0, 4, 2, 3))
                batch = torch.FloatTensor(batch)
                obs_out.data.copy_(batch)

                states = net.bs_prop(obs_in)
                obs_expectation = net.decoder(states.view(EP_LEN * BATCH_SIZE, -1))
                obs_expectation = obs_expectation.view(obs_in.size())
                err_pae = criterion_pae(obs_expectation, obs_out)
                postfix['pae valid loss'] = err_pae.data[0]

            if update % 500 == 0:
                recon_ims = obs_expectation.data.cpu().numpy()
                target_ims = obs_out.data.cpu().numpy()
                joint = np.concatenate((target_ims, recon_ims), axis=-2)
                my_utils.batch_to_sequence(joint, fpath='%s/valid_recon_%03d.gif' % (EXP_FOLDER, epoch))

            bar.set_postfix(**postfix)

        # do checkpointing
        torch.save(net.state_dict(), '%s/paegan_epoch_%d.pth' % (EXP_FOLDER, epoch))




def save_crisp_states():
    DATA_PARTITION_SIZE = 10

    try:
        os.makedirs(EXP_FOLDER)
        os.makedirs(DATA_FOLDER)
    except OSError:
        pass

    # prepare data
    data_train = DataContainer(
        '/home/ira/code/projects/nn-play/experiments/0__well_done/17-11-30_09:05-wp_1b_1l_small_deter/data/train.pt',
        batch_size=BATCH_SIZE, ep_len_read=EP_LEN)
    data_train.populate_images()

    pae = PredictiveAutoencoder(v_size=V_SIZE).cuda()
    pae.load_state_dict(torch.load("%s/autencoder_epoch_4.pth" % (EXP_FOLDER)))

    ims = data_train.images
    ims = ims.transpose((1, 0, 4, 2, 3))

    x = Variable(torch.FloatTensor(DATA_PARTITION_SIZE, *ims.shape[1:]).cuda())

    real_states = []
    for i in range(ims.shape[0] // DATA_PARTITION_SIZE):
        begin = i * DATA_PARTITION_SIZE
        end = (i+1) * DATA_PARTITION_SIZE
        im_slice = ims[begin:end, ...]

        x.data.copy_(torch.FloatTensor(im_slice))
        state_slice = pae.bs_prop(x).view((-1, BS_SIZE)).data.cpu().numpy()
        real_states.append(state_slice)

    real_states = np.concatenate(real_states, axis=0)
    np.save("{}/crisp_states.npy".format(DATA_FOLDER), real_states)


def train_GAN(start_at_epoch=0):

    # get data
    states = np.load("{}/crisp_states.npy".format(DATA_FOLDER))

    # prepare networks
    pae = PredictiveAutoencoder(v_size=V_SIZE)
    pae.load_state_dict(torch.load("%s/autencoder_epoch_4.pth" % (EXP_FOLDER)))
    net_decoder = pae.decoder.cuda()
    # net_GAN = BeliefStateGAN(bs_size=V_SIZE, n_size=N_SIZE, d_size=D_SIZE, g_size=G_SIZE)
    net_GAN = BeliefStateGAN().cuda()
    criterion = nn.BCELoss()

    noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE).cuda())
    fixed_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, N_SIZE).normal_(0, 1).cuda())
    bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, BS_SIZE).cuda())
    fixed_bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, BS_SIZE).uniform_(-1, 1).cuda())
    real = Variable(torch.FloatTensor(GAN_BATCH_SIZE, BS_SIZE).cuda())
    label = Variable(torch.FloatTensor(GAN_BATCH_SIZE).cuda())
    real_label = 1
    fake_label = 0

    optimiser_D = optim.Adam(net_GAN.D.parameters(), lr=0.0005)
    optimiser_G = optim.Adam(net_GAN.G.parameters(), lr=0.0005)

    start_at_epoch = 0
    # net_GAN.load_state_dict(torch.load("%s/gan_%d.pth" % (EXP_FOLDER, start_at_epoch-1)))

    for epoch in range(start_at_epoch, EPOCHS):
        for update in range(UPDATES_PER_EPOCH):

            draws = np.random.randint(states.shape[0], size=GAN_BATCH_SIZE)
            real_batch = torch.FloatTensor(states[draws, ...])

            net_GAN.D.zero_grad()

            real.data.copy_(real_batch)
            label.data.fill_(real_label)

            out_D = net_GAN.D(real)
            err_real = criterion(out_D, label)
            err_real.backward()
            D_x = out_D.data.mean()

            noise.data.normal_(0, 1)
            bs_noise.data.uniform_(-1, 1)
            fake = net_GAN.G(noise, bs_noise)
            label.data.fill_(fake_label)
            out_D = net_GAN.D(fake.detach())
            err_fake = criterion(out_D, label)
            err_fake.backward()
            D_G_z1 = out_D.data.mean()
            err_D = err_fake + err_real
            optimiser_D.step()

            net_GAN.G.zero_grad()
            label.data.fill_(real_label)
            out_D = net_GAN.D(fake)
            err_G = criterion(out_D, label)
            err_G.backward()
            D_G_z2 = out_D.data.mean()
            optimiser_G.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, update, UPDATES_PER_EPOCH,
                     err_D.data[0], err_G.data[0], D_x, D_G_z1, D_G_z2))
            if update % 100 == 0:
                # vutils.save_image(real_cpu,
                #         '%s/real_samples.png' % opt.outf,
                #         normalize=True)
                fake = net_GAN.G(fixed_noise, fixed_bs_noise)
                recon = net_decoder(fake)
                vutils.save_image(recon.data,
                        '%s/fake_samples_epoch_%03d.png' % (EXP_FOLDER, epoch),
                        normalize=False)

        # do checkpointing
        torch.save(net_GAN.state_dict(), '%s/gan_%d.pth' % (EXP_FOLDER, epoch))

if __name__ == "__main__":
    # test_autoencoder()
    # train_predictive_autoencoder()
    # train_GAN()
    train_PAEGAN()
    # save_crisp_states()
