import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time
import json
import functools 


def to_img_dict_(*inputs, super512=False):
    
    if type(inputs[0]) == tuple:
        inputs = inputs[0]
    res = {}
    res['output_64'] = inputs[0]
    res['output_128'] = inputs[1]
    res['output_256'] = inputs[2]
    # generator returns different things for 512PPAN
    if not super512:
        # from Generator
        mean_var = (inputs[3], inputs[4])
        loss = mean_var
    else:
        # from GeneratorL1Loss of 512PPAN
        res['output_512'] = inputs[3]
        l1loss = inputs[4] # l1 loss
        loss = l1loss

    return res, loss

def get_KL_Loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    discriminator_loss = real_d_loss + (wrong_d_loss+fake_d_loss) / 2.
    return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss  = criterion(real_img_logit, real_labels)
    fake_d_loss  = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2

def compute_g_loss(fake_logit, real_labels):

    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss

def compute_g_L2_loss(fake_image, real_image, coeff):
    criterion = nn.MSELoss(size_average=False)
    loss = criterion(fake_image, real_image) * coeff
    return loss

def compute_d_class_loss(image, class_id, coeff):
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(image, class_id.float()) * coeff
    return loss

def plot_imgs(samples,iter, epoch, typ, name, path='', model_name=None, plot=False):
    if name == 'test_samples':
        tmpX = save_images(samples, save=not path == '', save_path=os.path.join(
            path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
    elif name == 'train_samples':
        tmpX = save_images(samples, save=not path == '', save_path=os.path.join(
            path, '{}_iter{}_epoch{}_{}.png'.format(name, iter, epoch, typ)), dim_ordering='th')
    if plot:
        plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=model_name)

def one_hot(ids, class_num=0):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    batch_size = len(ids)
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids).view(-1, 1)
    out_tensor = torch.zeros(batch_size, class_num).scatter_(1, ids, 1.0)
    # out_tensor.scatter_(1, ids, 1.0)
    return out_tensor

def train_gans(dataset, model_root, model_name, netG, netD, vgg16, args):
    """
    Parameters:
    ----------
    dataset: 
        data loader. refers to fuel.dataset
    model_root: 
        the folder to save the model weights
    model_name : 
        the model_name 
    netG:
        Generator
    netD:
        Descriminator
    """

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' get train and test data sampler '''
    train_sampler = iter(dataset[0])
    test_sampler = iter(dataset[1])

    updates_per_epoch = int(dataset[0]._num_examples / args.batch_size)

    ''' configure optimizer '''
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=d_lr, betas=(0.5, 0.999))
    # optimizerD = optim.Adam([{'params': netD.pair_disc_256.linear.parameters()},
    #                          {'params': netD.pair_disc_256.class_node.parameters()}], lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #-------------load model from  checkpoint---------------------------#
    if args.reuse_weights:
        D_weightspath = os.path.join(
            model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(
            model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):
            weights_dict = torch.load(
                D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netD_.load_state_dict(weights_dict, strict=False)

            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(
                G_weightspath, map_location=lambda storage, loc: storage)
            netG_ = netG.module if 'DataParallel' in str(type(netG)) else netG
            netG_.load_state_dict(weights_dict, strict=False)

            # start_epoch = 1
            start_epoch = args.load_from_epoch + 1
            d_lr /= 2 ** (start_epoch // args.epoch_decay)
            g_lr /= 2 ** (start_epoch // args.epoch_decay) 
        else:
            raise ValueError('{} or {} do not exist'.format(D_weightspath, G_weightspath))
    else:
        start_epoch = 1

    #-------------init ploters for losses----------------------------#
    # d_loss_plot = plot_scalar(
    #     name="d_loss", env=model_name, rate=args.display_freq)
    # g_loss_plot = plot_scalar(
    #     name="g_loss", env=model_name, rate=args.display_freq)
    # lr_plot = plot_scalar(name="lr", env=model_name, rate=args.display_freq)
    # kl_loss_plot = plot_scalar(name="kl_loss", env=model_name, rate=args.display_freq)
    #
    # all_keys = ["output_64", "output_128", "output_256"]
    # g_plot_dict, g_plot_inception_dict, g_plot_l2_dict, g_plot_class_dict, d_plot_dict, d_plot_class_dict \
    #     = {}, {}, {}, {}, {}, {}
    # for this_key in all_keys:
    #     g_plot_dict[this_key] = plot_scalar(
    #         name="g_img_loss_" + this_key, env=model_name, rate=args.display_freq)
    #     g_plot_inception_dict[this_key] = plot_scalar(
    #         name="g_img_inception_loss_" + this_key, env=model_name, rate=args.display_freq)
    #     # g_plot_l2_dict[this_key] = plot_scalar(
    #     #     name="g_img_l2_loss_" + this_key, env=model_name, rate=args.display_freq)
    #     g_plot_class_dict[this_key] = plot_scalar(
    #         name="g_img_class_loss_" + this_key, env=model_name, rate=args.display_freq)
    #     d_plot_dict[this_key] = plot_scalar(
    #         name="d_img_loss_" + this_key, env=model_name, rate=args.display_freq)
    #     d_plot_class_dict[this_key] = plot_scalar(
    #         name="d_img_class_loss_" + this_key, env=model_name, rate=args.display_freq)

    #--------Generator niose placeholder used for testing------------#
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z)
    # generate a set of fixed test samples to visualize changes in training epoches
    fixed_images, _, fixed_embeddings, _, _, _, _ = next(test_sampler)
    fixed_embeddings = to_device(fixed_embeddings)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(
        0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a) for a in fixed_z_data]

    # create discrimnator label placeholder (not a good way)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(0)).cuda()

    def get_labels(logit):
        # get discriminator labels for real and fake
        if logit.size(-1) == 1: 
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)
    
    #--------Start training------------#
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        # reset to prevent StopIteration
        train_sampler = iter(dataset[0]) 
        test_sampler = iter(dataset[1])

        netG.train()
        netD.train()
        for it in range(updates_per_epoch):
            ncritic = args.ncritic

            for _ in range(ncritic):
                ''' Sample data '''
                try:
                    images, wrong_images, np_embeddings, _, _, np_class_ids, np_wrong_class_ids \
                                                                = next(train_sampler)
                except:
                    train_sampler = iter(dataset[0]) # reset
                    images, wrong_images, np_embeddings, _, _, np_class_ids, np_wrong_class_ids \
                                                                = next(train_sampler)
                # just classification not detection
                # no background, 0 is precise class information
                # np_class_ids -= 1
                # np_wrong_class_ids -=1
                embeddings = to_device(np_embeddings, requires_grad=False)
                # one_hot_class_ids = one_hot(np_class_ids, class_num=200).cuda() # 200 for birds, 102 for flowers, 91 for coco
                # one_hot_wrong_class_ids = one_hot(np_wrong_class_ids, class_num=200).cuda()
                # one_hot_class_ids = one_hot(np_class_ids, class_num=102).cuda()
                # one_hot_wrong_class_ids = one_hot(np_wrong_class_ids, class_num=102).cuda()
                # print(one_hot_wrong_class_ids)
                # _, indices = torch.max(one_hot_wrong_class_ids, 1)
                # print(indices)
                one_hot_class_ids = to_device(np_class_ids, requires_grad=False)
                one_hot_wrong_class_ids = to_device(np_wrong_class_ids, requires_grad=False)

                # class_ids = to_device(np_class_ids, requires_grad=False)
                # wrong_class_ids = to_device(np_wrong_class_ids, requires_grad=False)
                z.data.normal_(0, 1)

                ''' update D '''
                for p in netD.parameters(): p.requires_grad = True
                netD.zero_grad()

                fake_images, mean_var = to_img_dict(netG(embeddings, z))

                discriminator_loss = 0
                ''' iterate over image of different sizes.'''
                for key, _ in fake_images.items():
                    this_img = to_device(images[key])
                    this_wrong = to_device(wrong_images[key])
                    this_fake = Variable(fake_images[key].data)
                    if key != 'output_256':
                        real_logit,  real_img_logit_local = netD(this_img, embeddings)
                        wrong_logit, wrong_img_logit_local = netD(this_wrong, embeddings)
                        fake_logit,  fake_img_logit_local = netD(this_fake, embeddings)

                        ''' compute disc pair loss '''
                        real_labels, fake_labels = get_labels(real_logit)
                        pair_loss =  compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                        ''' compute disc image loss '''
                        real_labels, fake_labels = get_labels(real_img_logit_local)
                        img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local, real_labels, fake_labels)

                        discriminator_loss += (pair_loss + img_loss)

                        # d_plot_dict[key].plot(to_numpy(img_loss).mean())
                    else:
                        real_logit, real_img_logit_local, real_class_logit = netD(this_img, embeddings)
                        wrong_logit, wrong_img_logit_local, wrong_class_logit = netD(this_wrong, embeddings)
                        fake_logit, fake_img_logit_local, fake_class_logit = netD(this_fake, embeddings)

                        ''' compute disc pair loss '''
                        real_labels, fake_labels = get_labels(real_logit)
                        pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                        ''' compute disc image loss '''
                        real_labels, fake_labels = get_labels(real_img_logit_local)
                        img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local,
                                                      real_labels, fake_labels)

                        discriminator_loss += (pair_loss + img_loss)
                        # d_plot_dict[key].plot(to_numpy(img_loss).mean())

                        # train with wrong image, wrong label, real caption
                        d_lossC_wrong = compute_d_class_loss(wrong_class_logit, one_hot_wrong_class_ids, args.D_class_COE)

                        # train with real image, real label, real caption
                        d_lossC_real = compute_d_class_loss(real_class_logit, one_hot_class_ids, args.D_class_COE)

                        # train with fake image, real label, real caption
                        d_lossC_fake = compute_d_class_loss(fake_class_logit, one_hot_class_ids, args.D_class_COE)

                        d_lossC = d_lossC_wrong + d_lossC_real + d_lossC_fake
                        discriminator_loss += d_lossC
                        d_lossC_val = to_numpy(d_lossC).mean()
                        # d_plot_class_dict[key].plot(d_lossC_val)

                discriminator_loss.backward()
                optimizerD.step()
                netD.zero_grad()
                d_loss_val = to_numpy(discriminator_loss).mean()
                # d_loss_plot.plot(d_loss_val)

            ''' update G '''
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            # TODO Test if we do need to sample again in Birds and Flowers
            # z.data.normal_(0, 1)  # resample random noises
            # fake_images, kl_loss = netG(embeddings, z)

            loss_val = 0
            if type(mean_var) == tuple:
                kl_loss = get_KL_Loss(mean_var[0], mean_var[1])
                kl_loss_val = to_numpy(kl_loss).mean()
                generator_loss = args.KL_COE * kl_loss
            else:
                # when trian 512PPAN. KL loss is fixed since we assume 256PPAN is trained.
                # mean_var actually returns pixel-wise l1 loss (see paper)
                generator_loss = mean_var

            # kl_loss_plot.plot(kl_loss_val)
            #---- iterate over image of different sizes ----#
            '''Compute gen loss'''
            for key, _ in fake_images.items():

                this_fake = fake_images[key].cuda()
                if key != 'output_256':
                    fake_pair_logit, fake_img_logit_local = netD(this_fake, embeddings)

                    # -- compute pair loss ---
                    real_labels, _ = get_labels(fake_pair_logit)
                    generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                    # -- compute image loss ---
                    real_labels, _ = get_labels(fake_img_logit_local)
                    img_loss = compute_g_loss(fake_img_logit_local, real_labels)
                    generator_loss += img_loss
                    # g_plot_dict[key].plot(to_numpy(img_loss).mean())

                else:
                    fake_pair_logit, fake_img_logit_local, fake_class_logit = netD(this_fake, embeddings)

                    # -- compute pair loss ---
                    real_labels, _ = get_labels(fake_pair_logit)
                    generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                    # -- compute image loss ---
                    real_labels, _ = get_labels(fake_img_logit_local)
                    img_loss = compute_g_loss(fake_img_logit_local, real_labels)
                    generator_loss += img_loss
                    # g_plot_dict[key].plot(to_numpy(img_loss).mean())

                    # -- compute perceptual loss for output_256 ---
                    this_img = to_device(images[key], requires_grad=False)
                    this_feature = vgg16(this_img)
                    fake_feature = vgg16(this_fake)
                    g_ince_loss = compute_g_L2_loss(fake_feature, this_feature, args.G_ince_COE)
                    g_ince_loss_val = to_numpy(g_ince_loss).mean()
                    generator_loss += g_ince_loss
                    # g_plot_inception_dict[key].plot(g_ince_loss_val)

                    # train with fake image, real label, real caption
                    g_lossC_fake = compute_d_class_loss(fake_class_logit, one_hot_class_ids, args.D_class_COE)
                    g_lossC_fake_val = to_numpy(g_lossC_fake).mean()
                    # g_plot_class_dict[key].plot(g_lossC_fake_val)
                    generator_loss += g_lossC_fake

            generator_loss.backward()
            optimizerG.step()
            netG.zero_grad()
            g_loss_val = to_numpy(generator_loss).mean()
            # g_loss_plot.plot(g_loss_val)
            # lr_plot.plot(g_lr)

            # --- visualize train samples----
            if it % args.verbose_per_iter == 0:
                # print('[epoch %d/%d iter %d/%d]: lr = %.6f g_loss = %.5f d_loss= %.5f' %
                #       (epoch, tot_epoch, it, updates_per_epoch, g_lr, g_loss_val, d_loss_val))
                print('[epoch %d/%d iter %d/%d]: lr = %.10f g_loss = %.5f '
                      'g_ince_loss = %.5f g_class_loss = %.5f d_class_loss = %.5f d_loss = %.5f' %
                      (epoch, tot_epoch, it, updates_per_epoch, g_lr, g_loss_val,
                       g_ince_loss_val, g_lossC_fake_val, d_lossC_val, d_loss_val-d_lossC_val))
                sys.stdout.flush()

            if it % args.display_train_imgs == 0:
                for k, sample in fake_images.items():
                    plot_imgs(samples=[to_numpy(images[k]), to_numpy(sample)], iter=it,
                              epoch=epoch, typ=k, name='train_samples', path=model_folder, model_name=model_name)

        # generate and visualize testing results per epoch
        # display original image and the sampled images
        if epoch % args.display_test_imgs == 0:
            vis_samples = {}
            for idx_test in range(2):
                if idx_test == 0:
                    test_images, test_embeddings = fixed_images, fixed_embeddings
                else:
                    test_images, _, test_embeddings, _, _, _, _ = next(test_sampler)
                    test_embeddings = to_device(test_embeddings, volatile=True)
                    testing_z = Variable(z.data, volatile=True)
                for t in range(args.test_sample_num):
                    if idx_test == 0:
                        testing_z = fixed_z_list[t]
                    else:
                        testing_z.data.normal_(0, 1)
                    fake_images, _ = to_img_dict(netG(test_embeddings, testing_z))
                    samples = fake_images
                    if idx_test == 0 and t == 0:
                        for k in samples.keys():
                            #  +1 to make space for real image
                            vis_samples[k] = [None for i in range(
                                args.test_sample_num + 1)]

                    for k, v in samples.items():
                        cpu_data = to_numpy(v)
                        if t == 0:
                            if vis_samples[k][0] is None:
                                vis_samples[k][0] = test_images[k]
                            else:
                                vis_samples[k][0] = np.concatenate([vis_samples[k][0], test_images[k]], 0)

                        if vis_samples[k][t+1] is None:
                            vis_samples[k][t+1] = cpu_data
                        else:
                            vis_samples[k][t+1] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)
            # visualize testing samples
            for typ, v in vis_samples.items():
                plot_imgs(samples=v, iter=0, epoch=epoch, typ=k, name='test_samples',
                          path=model_folder, model_name=model_name, plot=False)

        ''' save weights '''
        if epoch % args.save_freq == 0:
            netD = netD.cpu()
            netG = netG.cpu()
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netG_ = netG.module if 'DataParallel' in str(type(netD)) else netG
            torch.save(netD_.state_dict(), os.path.join(
                model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG_.state_dict(), os.path.join(
                model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            netD = netD.cuda()
            netG = netG.cuda()
        end_timer = time.time() - start_timer
        print(
            'epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))
