#!/usr/bin/env python3

import tqdm
import argparse
import os
import torch
import torch.optim as optim
import torchvision.utils as vutils

import torch_utils
from structured_container import DataContainer
from torch_nets import *
import torch_nets

PAE_BATCH_SIZE = 4
GAN_BATCH_SIZE = 16
AVERAGING_BATCH_SIZE = 32
EP_LEN = 100

AVERAGING_ERROR_MULTIPLIER = 1000

BALLS_OBS_SHAPE = (1, 28, 28)

GUARANTEED_PERCEPTS = 6
UNCERTAIN_PERCEPTS = 4
P_NO_OBS_VALID = 1.0

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
    parser.add_argument('--updates_per_epoch', default=3000, type=int,
                        help="How many updates per epoch?")
    parser.add_argument('--training_stage', type=str,
                        choices=['pae', 'paegan', 'paegan-sampler', 'sampler', 'averager', 'd-sampler'],
                        help="Different training modes enable training of different parts of the network")
    parser.add_argument('--p_mask', default=0.99, type=float,
                        help="What fraction of input observations is masked? eg 0.6")
    parser.add_argument('--compare_with_pf', default=False, type=bool,
                        help="Should the results be compared with particle filter?")
    parser.add_argument('--cuda', default=True, type=bool,
                        help="Should CUDA be used?")

    args = parser.parse_args()
    parser.print_help()
    print(args)

    output_dir = args.output_dir
    n_epochs = args.epochs
    updates_per_epoch = args.updates_per_epoch
    training_stage = args.training_stage
    use_cuda = args.cuda
    p_mask = args.p_mask
    train_d_every_n_updates = 1

    if training_stage == "pae":
        train_pae_switch = True
        train_d_switch = False
        train_g_switch = False
        train_av_switch = False
    elif training_stage == "paegan":
        train_pae_switch = False
        train_d_switch = True
        train_g_switch = True
        train_av_switch = True
    elif training_stage == "d-sampler":
        train_pae_switch = False
        train_d_switch = True
        train_g_switch = True
        train_av_switch = True
    elif training_stage == "sampler":
        train_d_every_n_updates = 5
        train_pae_switch = False
        train_d_switch = True
        train_g_switch = True
        train_av_switch = True
    elif training_stage == "averager":
        train_pae_switch = False
        train_d_switch = False
        train_g_switch = False
        train_av_switch = True
    else:
        raise ValueError('Wrong training stage {}'.format(training_stage))

    if use_cuda:
        assert torch.cuda.is_available() is True

    if not os.path.exists(args.output_dir):
        torch_utils.make_dir_tree(args.output_dir)

    # prepare data
    sim_config = None
    obs_shape = None
    train_getter = None
    valid_getter = None
    if args.dataset_type == 'balls':
        sim_config = torch.load('{}/train.conf'.format(args.data_dir))
        obs_shape = BALLS_OBS_SHAPE

        train_container = DataContainer('{}/train.pt'.format(args.data_dir), batch_size=PAE_BATCH_SIZE)
        valid_container = DataContainer('{}/valid.pt'.format(args.data_dir), batch_size=PAE_BATCH_SIZE)
        sim_config = torch.load(open('{}/train.conf'.format(args.data_dir), 'rb'))

        train_container.populate_images()
        valid_container.populate_images()

        train_getter = train_container.get_batch_episodes
        valid_getter = valid_container.get_batch_episodes

    else:
        raise ValueError('Failed to load data. Wrong dataset type {}'.format(args.dataset_type))

    if args.compare_with_pf:
        assert sim_config is not None

    # prepare network
    if args.dataset_type == 'balls':
        net = PAEGAN()
        noise_size = torch_nets.N_SIZE
        bs_size = torch_nets.BS_SIZE

    else:
        raise ValueError('Failed to initialise model. Wrong dataset type {}'.format(args.dataset_type))

    if args.start_from_checkpoint is not None:
        assert type(args.start_from_checkpoint) == int
        net.load_state_dict(torch.load("{}/network/paegan_epoch_{}.pth".format(args.output_dir, args.start_from_checkpoint)))
        current_epoch = args.start_from_checkpoint + 1
    else:
        current_epoch = 0

    # build report file

    # initialise variables
    real_label = 1
    fake_label = 0
    if use_cuda:
        net = net.cuda()
        criterion_pae = nn.MSELoss().cuda()
        criterion_gan = nn.BCELoss().cuda()
        # criterion_gan = nn.MSELoss().cuda()
        criterion_gen_averaged = nn.MSELoss().cuda()

        obs_in = Variable(torch.FloatTensor(EP_LEN, PAE_BATCH_SIZE, *BALLS_OBS_SHAPE).cuda())
        obs_out = Variable(torch.FloatTensor(EP_LEN, PAE_BATCH_SIZE, *BALLS_OBS_SHAPE).cuda())

        averaging_noise = Variable(torch.FloatTensor(AVERAGING_BATCH_SIZE, noise_size).cuda())
        g_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, noise_size).cuda())

        fixed_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, noise_size).normal_(0, 1).cuda())
        fixed_bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, bs_size).uniform_(-1, 1).cuda())
        fake_labels = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1).cuda())
        real_labels = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1).cuda())
    else:
        criterion_pae = nn.MSELoss()
        criterion_gan = nn.BCELoss()
        # criterion_gan = nn.MSELoss()
        criterion_gen_averaged = nn.MSELoss()

        obs_in = Variable(torch.FloatTensor(EP_LEN, PAE_BATCH_SIZE, *BALLS_OBS_SHAPE))
        obs_out = Variable(torch.FloatTensor(EP_LEN, PAE_BATCH_SIZE, *BALLS_OBS_SHAPE))

        averaging_noise = Variable(torch.FloatTensor(AVERAGING_BATCH_SIZE, noise_size))
        g_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, noise_size))

        fixed_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, noise_size).normal_(0, 1))
        fixed_bs_noise = Variable(torch.FloatTensor(GAN_BATCH_SIZE, bs_size).uniform_(-1, 1))
        fake_labels = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1))
        real_labels = Variable(torch.FloatTensor(GAN_BATCH_SIZE, 1))

    # optimisers
    optimiser_pae = optim.Adam([{'params': net.bs_prop.parameters()},
                                {'params': net.decoder.parameters()}],
                               lr=0.0003)
    # optimiser_g = optim.Adam([{'params': net.bs_prop.parameters(), 'lr': 0.00005},
    #                           {'params': net.G.parameters(), 'lr': 0.0002}])
    optimiser_g = optim.Adam(net.G.parameters(), lr=0.0002)
    optimiser_d = optim.Adam(net.D.parameters(), lr=0.0002)

    # start training
    epoch_report = {}
    until_epoch = current_epoch + n_epochs
    for current_epoch in range(current_epoch, until_epoch):
        torch_utils.pf_comparison(net, sim_config, output_dir, current_epoch)

        bar = tqdm.trange(updates_per_epoch)
        epoch_report['epoch'] = '[{}/{}]'.format(current_epoch, until_epoch)

        for update in bar:
            net.zero_grad()
            losses = []

            batch = train_getter()
            masked = torch_utils.mask_percepts(batch, p=p_mask)

            batch = batch.transpose((1, 0, 4, 2, 3))
            masked = masked.transpose((1, 0, 4, 2, 3))

            batch = torch.FloatTensor(batch)
            masked = torch.FloatTensor(masked)

            obs_in.data.copy_(masked)
            obs_out.data.copy_(batch)

            # generate beliefs states
            # _ep means tensor has shape (ep_len, batch_size, *obs_shape)
            # _nonep means tensor has shape (ep_len * batch_size, *obs_shape)
            states_ep = net.bs_prop(obs_in)
            states_nonep = states_ep.view(EP_LEN * PAE_BATCH_SIZE, -1)

            obs_expectation = None
            if train_av_switch and not train_pae_switch:
                obs_expectation = net.decoder(states_nonep).view(obs_in.size())

            elif train_pae_switch is True:
                obs_expectation = net.decoder(states_nonep).view(obs_in.size())
                err_pae = criterion_pae(obs_expectation, obs_out)
                losses.append(err_pae)
                epoch_report['pae train loss'] = err_pae.data[0]

            if train_d_switch is True and update % train_d_every_n_updates == 0:
                real_labels.data.fill_(real_label)
                fake_labels.data.fill_(fake_label)

                obs_out_nonep = obs_out.view(EP_LEN * PAE_BATCH_SIZE,
                                             obs_out.size(2), obs_out.size(3), obs_out.size(4))
                # draw real observations for D training
                draw = np.random.choice(EP_LEN * PAE_BATCH_SIZE, size=GAN_BATCH_SIZE, replace=False)
                obs_d = obs_out_nonep[draw, ...]

                # draw states for D training
                draw = np.random.choice(EP_LEN * PAE_BATCH_SIZE, size=GAN_BATCH_SIZE, replace=False)
                states_d = states_nonep[draw, ...]

                # train discriminator with real data
                out_d_real = net.D(obs_d).view(GAN_BATCH_SIZE, 1)
                # print("out_d_real", out_d_real)
                err_d_real = criterion_gan(out_d_real, real_labels)

                # train discriminator with fake data
                g_noise.data.normal_(0, 1)
                state_sample = net.G(g_noise, states_d)
                obs_sample = net.decoder(state_sample)
                out_d_fake = net.D(obs_sample.detach()).view(GAN_BATCH_SIZE, 1)
                # print("out_d_fake", out_d_fake)
                err_d_fake = criterion_gan(out_d_fake, fake_labels)

                err_d = (err_d_fake + err_d_real) / 2
                # losses.append(err_d)

                err_d.backward()
                optimiser_d.step()

                epoch_report['d train loss'] = err_d.data[0]

                if update == 0:
                    vutils.save_image(obs_d.data,
                                      '{}/images/real_samples.png'.format(output_dir),
                                      normalize=True)

            if train_g_switch is True:
                # train generator using discriminator
                # draw states for G training
                draw = np.random.choice(EP_LEN * PAE_BATCH_SIZE, size=GAN_BATCH_SIZE, replace=False)
                states_g = states_nonep[draw, ...]

                g_noise.data.normal_(0, 1)
                state_sample = net.G(g_noise, states_g.detach())
                obs_sample = net.decoder(state_sample)

                out_d_g = net.D(obs_sample).view(GAN_BATCH_SIZE, 1)
                # print("out_d_g", out_d_g)
                err_g = criterion_gan(out_d_g, real_labels)
                losses.append(err_g)

                epoch_report['g train loss'] = err_g.data[0]

                if update % 100 == 0:
                    state_sample = net.G(fixed_noise, states_g)
                    obs_sample = net.decoder(state_sample)
                    vutils.save_image(obs_sample.data,
                                      '{}/images/fake_samples_epoch_{}.png'.format(output_dir, current_epoch),
                                      normalize=False)

            if train_av_switch is True:
                # train generator using averaging
                # draw random states
                draw = np.random.choice(EP_LEN * PAE_BATCH_SIZE, size=1, replace=False)
                states_av = states_nonep[draw, ...]
                states_av = states_av.expand(AVERAGING_BATCH_SIZE, -1)

                # get corresponding observation expectation
                obs_exp_nonep = obs_expectation.view(EP_LEN * PAE_BATCH_SIZE,
                                                     obs_out.size(2), obs_out.size(3), obs_out.size(4))
                obs_exp = obs_exp_nonep[draw, ...]

                # generate samples from state
                averaging_noise.data.normal_(0, 1)
                n_samples = net.G(averaging_noise, states_av.detach())

                n_samples = net.decoder(n_samples)
                # print('samples size', n_samples.size())

                sample_av = n_samples.mean(dim=0).unsqueeze(0)

                err_av = criterion_gen_averaged(sample_av, obs_exp.detach())

                # normalise error to ~1

                losses.append(AVERAGING_ERROR_MULTIPLIER * err_av)
                epoch_report['av train loss'] = err_av.data[0]

                if update % 10 == 0:
                    sample_mixture = sample_av.data.cpu().numpy()
                    observation_belief = obs_exp.data.cpu().numpy()
                    joint = np.concatenate((observation_belief, sample_mixture), axis=-2)
                    joint = np.expand_dims(joint, axis=0)
                    torch_utils.batch_to_sequence(joint, fpath='{}/images/sum_{}.gif'.format(output_dir, current_epoch))

            # =====================================
            # UPDATE WEIGHTS HERE!
            if len(losses) > 0:
                sum(losses).backward()

            if train_pae_switch:
                optimiser_pae.step()

            if train_g_switch or train_av_switch:
                optimiser_g.step()

            # pae validation error and image record
            if update % 100 == 0:
                batch = valid_getter()
                masked = torch_utils.mask_percepts(batch, p=p_mask)

                batch = batch.transpose((1, 0, 4, 2, 3))
                masked = masked.transpose((1, 0, 4, 2, 3))

                batch = torch.FloatTensor(batch)
                masked = torch.FloatTensor(masked)

                obs_in.data.copy_(masked)
                obs_out.data.copy_(batch)

                # generate beliefs states
                states_ep = net.bs_prop(obs_in)
                states_nonep = states_ep.view(EP_LEN * PAE_BATCH_SIZE, -1)

                obs_expectation = net.decoder(states_nonep).view(obs_in.size())
                err_valid_pae = criterion_pae(obs_expectation, obs_out)
                epoch_report['pae valid loss'] = err_valid_pae.data[0]

                # print a gif
                if update % 500 == 0:
                    recon_ims = obs_expectation.data.cpu().numpy()
                    target_ims = obs_out.data.cpu().numpy()
                    joint = np.concatenate((target_ims, recon_ims), axis=-2)
                    torch_utils.batch_to_sequence(joint, fpath='{}/images/valid_recon_{}.gif'.format(output_dir, current_epoch))

            bar.set_postfix(**epoch_report)

        torch.save(net.state_dict(), '{}/network/paegan_epoch_{}.pth'.format(output_dir, current_epoch))
