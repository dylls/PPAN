# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from ..proj_utils.network_utils import *
import math
import functools
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply_layer(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

    @staticmethod
    def apply(module, name):
        conv_class_name = torch.nn.modules.conv.Conv2d
        linear_class_name = torch.nn.Linear
        if isinstance(module, conv_class_name) or isinstance(module, linear_class_name):
            SpectralNorm.apply_layer(module, name)
        else:
            for i in range(0, len(module)):
                if isinstance(module[i], conv_class_name) or isinstance(module[i], linear_class_name):
                    SpectralNorm.apply_layer(module[i], name)


    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module

class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()

        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear = nn.Linear(noise_dim, emb_dim*2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma, kl_loss=False):
    
        epsilon = Variable(torch.cuda.FloatTensor(mean.size()).normal_())
        stddev  = logsigma.exp()
        
        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs, kl_loss=True):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma)
        return c, mean, log_sigma

#-----------------------------------------------#
#    used to encode image into feature maps     #
#-----------------------------------------------#

class ImageDown(torch.nn.Module):

    def __init__(self, input_size, num_chan, out_dim):
        """
            Parameters:
            ----------
            input_size: int
                input image size, can be 64, or 128, or 256
            num_chan: int
                channel of input images.
            out_dim : int
                the dimension of generated image code.
        """

        super(ImageDown, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)

        _layers = []
        # use large kernel_size at the end to prevent using zero-padding and stride
        if input_size == 64:
            cur_dim = 128
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 32
            _layers += [conv_norm(cur_dim, cur_dim*2, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 128:
            cur_dim = 64
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 64
            _layers += [conv_norm(cur_dim, cur_dim*2, norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(cur_dim*2, cur_dim*4, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 256:
            cur_dim = 32 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 128
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 64
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 32
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=2, activation=activ)] # 8

        # if input_size == 64:
        #     cur_dim = 128
        #     _layers = [spectral_norm(conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False))]  # 32
        #     _layers += [spectral_norm(conv_norm(cur_dim, cur_dim * 2, norm_layer, stride=2, activation=activ, use_norm=False))]  # 16
        #     _layers += [spectral_norm(conv_norm(cur_dim * 2, cur_dim * 4, norm_layer, stride=2, activation=activ, use_norm=False))]  # 8
        #     _layers += [
        #         spectral_norm(conv_norm(cur_dim * 4, out_dim, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0, use_norm=False))]  # 4
        #
        # if input_size == 128:
        #     cur_dim = 64
        #     _layers += [spectral_norm(conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False))]  # 64
        #     _layers += [spectral_norm(conv_norm(cur_dim, cur_dim * 2, norm_layer, stride=2, activation=activ, use_norm=False))]  # 32
        #     _layers += [spectral_norm(conv_norm(cur_dim * 2, cur_dim * 4, norm_layer, stride=2, activation=activ, use_norm=False))]  # 16
        #     _layers += [spectral_norm(conv_norm(cur_dim * 4, cur_dim * 8, norm_layer, stride=2, activation=activ, use_norm=False))]  # 8
        #     _layers += [
        #         spectral_norm(conv_norm(cur_dim * 8, out_dim, norm_layer, stride=1, activation=activ, kernel_size=5,
        #                                 padding=0, use_norm=False))]  # 4
        #
        # if input_size == 256:
        #     cur_dim = 32  # for testing
        #     _layers += [spectral_norm(conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False))]  # 128
        #     _layers += [spectral_norm(conv_norm(cur_dim, cur_dim * 2, norm_layer, stride=2, activation=activ, use_norm=False))]  # 64
        #     _layers += [spectral_norm(conv_norm(cur_dim * 2, cur_dim * 4, norm_layer, stride=2, activation=activ, use_norm=False))]  # 32
        #     _layers += [spectral_norm(conv_norm(cur_dim * 4, cur_dim * 8, norm_layer, stride=2, activation=activ, use_norm=False))]  # 16
        #     _layers += [spectral_norm(conv_norm(cur_dim * 8, out_dim, norm_layer, stride=2, activation=activ, use_norm=False))]  # 8

        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):

        out = self.node(inputs)
        return out


class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, kernel_size, this_size):
        """
            Parameters:
            ----------
            enc_dim: int
                the channel of image code.
            emb_dim: int
                the channel of sentence code.
            kernel_size : int
                kernel size used for final convolution.
        """

        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)
        inp_dim = enc_dim + emb_dim
        self.size = this_size
        # _layers = [spectral_norm(conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1,
        #                                    activation=activ, use_norm=False))]
        # _pair_layers = _layers + [spectral_norm(nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True))]
        _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ)]
        _pair_layers = _layers + [nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True)]
        self.pair_node = nn.Sequential(*_pair_layers)

        # _class_layers = _layers + [spectral_norm(nn.Conv2d(enc_dim, enc_dim, kernel_size=kernel_size, padding=0, bias=True))]
        _class_layers = _layers + [nn.Conv2d(enc_dim, enc_dim, kernel_size=kernel_size, padding=0, bias=True)]
        self.class_node = nn.Sequential(*_class_layers)
        # self.linear = spectral_norm(nn.Linear(enc_dim, 200)) # 200 for birds, 102 for flowers， 91for coco
        # self.linear = nn.Linear(enc_dim, 200)
        # self.linear = nn.Linear(enc_dim, 102)
        self.linear = nn.Linear(enc_dim, 91)


    def forward(self, sent_code,  img_code):

        sent_code = sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        dst_shape[1] = sent_code.size()[1]
        dst_shape[2] = img_code.size()[2]
        dst_shape[3] = img_code.size()[3]
        sent_code = sent_code.expand(dst_shape)
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        if self.size != 256:
            pair_output = self.pair_node(comp_inp)
            chn = pair_output.size()[1]
            pair_output = pair_output.view(-1, chn)
            return pair_output
        else:
            # print(comp_inp.shape)
            pair_output = self.pair_node(comp_inp)
            chn = pair_output.size()[1]
            pair_output = pair_output.view(-1, chn)

            class_output = self.class_node(comp_inp)
            # print(class_output.shape)
            class_output = class_output.squeeze(-1).squeeze(-1)
            # print(class_output.shape)
            class_output = self.linear(class_output)
            # class_output = class_output.view(-1, 200) # 200 for birds, 102 for flowers, 91 for coco
            # class_output = class_output.view(-1, 102)
            class_output = class_output.view(-1, 91)
            return pair_output, class_output

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.ReLU(True)

        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ),
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        return self.res_block(input) + input


class Sent2FeatMap(nn.Module):
    # used to project a sentence code into a set of feature maps
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


#----------------Define Generator and Discriminator-----------------------------#

class Generator(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, num_resblock=1, side_output_at=[64, 128, 256]):
        """
        Parameters:
        ----------
        sent_dim: int
            the dimension of sentence embedding
        noise_dim: int
            the dimension of noise input
        emb_dim : int
            the dimension of compressed sentence embedding.
        hid_dim: int
            used to control the number of feature maps.
        num_resblock: int
            the scale factor of generator (see paper for explanation).
        side_output_at:  list
            contains local loss size for discriminator at scales.
        """

        super(Generator, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        act_layer = nn.ReLU(True)
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)

        self.vec_to_tensor = Sent2FeatMap(
            emb_dim+noise_dim, 4, 4, self.hid_dim*8)
        self.side_output_at = side_output_at
        # feature map dimension reduce at which resolution
        reduce_dim_at = [8, 32, 128, 256]
        # different scales for all networks
        num_scales = [4, 8, 16, 32, 64, 128, 256]
        upsample_scales = [32, 64, 128]

        cur_dim = self.hid_dim*8
        for i in range(len(num_scales)):
            seq = []
            # unsampling
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            # if need to reduce dimension
            if num_scales[i] in reduce_dim_at:
                seq += [pad_conv_norm(cur_dim, cur_dim//2,
                                      norm_layer, activation=act_layer)]
                cur_dim = cur_dim//2
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim)]
            # add main convolutional module
            setattr(self, 'scale_%d' % (num_scales[i]), nn.Sequential(*seq))

            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d' %
                        (num_scales[i]), branch_out(in_dim=cur_dim))
                setattr(self, 'img_%d_add_fuse' %
                        (num_scales[i]), fuse_conv(in_dim=cur_dim))

            if num_scales[i] in upsample_scales:
                if i != len(num_scales) - 1 and (num_scales[i + 1] in reduce_dim_at):
                    setattr(self, 'img_%d_upsample_%d' %
                            (num_scales[i], num_scales[i] * 2), branch_upsample(in_dim=cur_dim, out_dim=cur_dim // 2))
                else:
                    setattr(self, 'img_%d_upsample_%d' %
                            (num_scales[i], num_scales[i] * 2), branch_upsample(in_dim=cur_dim, out_dim=cur_dim))

        self.apply(weights_init)
        print('>> Init PPAN Generator')
        print('\t side output at {}'.format(str(side_output_at)))

    def forward(self, sent_embeddings, z):
        """
        Parameters:
        ----------
        sent_embeddings: [B, sent_dim]
            sentence embedding obtained from char-rnn
        z: [B, noise_dim]
            noise input

        Returns:
        ----------
        out_dict: dictionary
            dictionary containing the generated images at scale [64, 128, 256]
        kl_loss: tensor
            Kullback–Leibler divergence loss from conditionining embedding
        """
        # sent_num = 10
        # print('sent_num: ', sent_num)
        # sent_random = 0
        # for _ in range(sent_num):
        #     sent_random_i, mean, logsigma=self.condEmbedding(sent_embeddings)
        #     sent_random += sent_random_i
        # sent_random /= sent_num

        sent_random, mean, logsigma = self.condEmbedding(sent_embeddings)

        text = torch.cat([sent_random, z], dim=1)
        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)

        # # skip 4x4 feature map to 32 and send to 64
        # x_64 = self.scale_64(x_32)
        # output_64 = self.tensor_to_img_64(x_64)
        #
        # # skip 8x8 feature map to 64 and send to 128
        # x_128 = self.scale_128(x_64)
        # output_128 = self.tensor_to_img_128(x_128)
        #
        # # skip 16x16 feature map to 128 and send to 256
        # out_256 = self.scale_256(x_128)
        # self.keep_out_256 = out_256
        # output_256 = self.tensor_to_img_256(out_256)

        output_32_64 = self.img_32_upsample_64(x_32)

        # skip 4x4 feature map to 32 and send to 64
        x_64 = self.scale_64(x_32)

        output_64 = self.img_64_add_fuse(x_64 + output_32_64)
        output_64_128 = self.img_64_upsample_128(output_64)
        output_64 = self.tensor_to_img_64(output_64)

        # skip 8x8 feature map to 64 and send to 128
        x_128 = self.scale_128(x_64)
        output_128 = self.img_128_add_fuse(x_128 + output_64_128)
        output_128_256 = self.img_128_upsample_256(output_128)
        output_128 = self.tensor_to_img_128(output_128)

        # skip 16x16 feature map to 128 and send to 256
        x_256 = self.scale_256(x_128)
        self.keep_out_256 = x_256
        output_256 = self.img_256_add_fuse(x_256 + output_128_256)
        # class_256 = self.linear(output_256)
        output_256 = self.tensor_to_img_256(output_256)

        return output_64, output_128, output_256, mean, logsigma


class Discriminator(torch.nn.Module):
    def __init__(self, num_chan,  hid_dim, sent_dim, emb_dim, side_output_at=[64, 128, 256]):
        """
        Parameters:
        ----------
        num_chan: int
            channel of generated images.
        enc_dim: int
            Reduce images inputs to (B, enc_dim, H, W) feature
        emb_dim : int
            the dimension of compressed sentence embedding.
        side_output_at:  list
            contains local loss size for discriminator at scales.
        """

        super(Discriminator, self).__init__()
        self.__dict__.update(locals())

        activ = nn.LeakyReLU(0.2, True)
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        self.side_output_at = side_output_at

        enc_dim = hid_dim * 4  # the ImageDown output dimension

        if 64 in side_output_at:  # discriminator for 64 input
            self.img_encoder_64 = ImageDown(64,  num_chan,  enc_dim)  # 4x4
            self.pair_disc_64 = DiscClassifier(enc_dim, emb_dim, kernel_size=4, this_size=64)
            # self.local_img_disc_64 = spectral_norm(nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True))  # 1x1
            # _layers = [spectral_norm(nn.Linear(sent_dim, emb_dim)), activ]
            self.local_img_disc_64 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)  # 1x1
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)

        if 128 in side_output_at:  # discriminator for 128 input
            self.img_encoder_128 = ImageDown(128,  num_chan, enc_dim)  # 4x4
            self.pair_disc_128 = DiscClassifier(enc_dim, emb_dim, kernel_size=4, this_size=128)
            # self.local_img_disc_128 = spectral_norm(nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True))  # 1x1
            # # map sentence to a code of length emb_dim
            # _layers = [spectral_norm(nn.Linear(sent_dim, emb_dim)), activ]
            self.local_img_disc_128 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)  # 1x1
            # map sentence to a code of length emb_dim
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)

        if 256 in side_output_at:  # discriminator for 256 input
            self.img_encoder_256 = ImageDown(256, num_chan, enc_dim)     # 8x8
            # self.pre_encode = spectral_norm(conv_norm(enc_dim, enc_dim, norm_layer, stride=1, activation=activ, kernel_size=5,
            #                             padding=0, use_norm=False))                       # 4x4
            self.pre_encode = conv_norm(enc_dim, enc_dim, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)
            self.pair_disc_256 = DiscClassifier(enc_dim, emb_dim, kernel_size=4, this_size=256)
            # never use 1x1 convolutions as the image disc classifier
            # self.local_img_disc_256 = spectral_norm(nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True))    # 5x5
            self.local_img_disc_256 = nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)
            # map sentence to a code of length emb_dim
            # _layers = [spectral_norm(nn.Linear(sent_dim, emb_dim)), activ]
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)

        print('>> Init PPAN Discriminator')
        print('\t Add adversarial loss at scale {}'.format(str(side_output_at)))

    def forward(self, images, embedding):
        '''
        Parameters:
        -----------
        images:    (B, C, H, W)
            input image tensor
        embedding : (B, sent_dim)
            corresponding embedding
        outptuts:  
        -----------
        out_dict: dict
            dictionary containing: pair discriminator output and image discriminator output
        '''
        out_dict = OrderedDict()
        this_img_size = images.size()[3]
        assert this_img_size in [32, 64, 128, 256], 'wrong input size {} in image discriminator'.format(this_img_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(this_img_size))
        local_img_disc = getattr(
            self, 'local_img_disc_{}'.format(this_img_size), None)
        pair_disc = getattr(self, 'pair_disc_{}'.format(this_img_size))
        context_emb_pipe = getattr(
            self, 'context_emb_pipe_{}'.format(this_img_size))

        sent_code = context_emb_pipe(embedding)
        img_code = img_encoder(images)

        if this_img_size == 256:
            pre_img_code = self.pre_encode(img_code)
            pair_disc_out, class_disc_out = pair_disc(sent_code, pre_img_code)
            local_img_disc_out = local_img_disc(img_code)
            return pair_disc_out, local_img_disc_out, class_disc_out
        else:
            pair_disc_out = pair_disc(sent_code, img_code)
            local_img_disc_out = local_img_disc(img_code)
            return pair_disc_out, local_img_disc_out



class GeneratorSuperL1Loss(nn.Module):
    # for 512 resolution
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, G256_weightspath='', num_resblock=2):

        super(GeneratorSuperL1Loss, self).__init__()
        self.__dict__.update(locals())
        print('>> Init a PPAN 512Generator (resblock={})'.format(num_resblock))
        
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        act_layer = nn.ReLU(True)
        
        self.generator_256 = Generator(sent_dim, noise_dim, emb_dim, hid_dim)
        if G256_weightspath != '':
            print ('pre-load generator_256 from', G256_weightspath)
            weights_dict = torch.load(G256_weightspath, map_location=lambda storage, loc: storage)
            self.generator_256.load_state_dict(weights_dict)

        # puch it to every high dimension
        scale = 512
        cur_dim = 64
        seq = []
        for i in range(num_resblock):
            seq += [ResnetBlock(cur_dim)]
        
        seq += [nn.Upsample(scale_factor=2, mode='nearest')]
        seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=act_layer)]
        cur_dim = cur_dim // 2

        setattr(self, 'scale_%d'%(scale), nn.Sequential(*seq) )
        setattr(self, 'tensor_to_img_%d'%(scale), branch_out(cur_dim))
        self.apply(weights_init)

    def parameters(self):
        
        fixed = list(self.generator_256.parameters())
        all_params = list(self.parameters())
        partial_params = list(set(all_params) - set(fixed))

        print ('WARNING: fixed params {} training params {}'.format(len(fixed), len(partial_params)))
        print('          It needs modifications if you can train all from scratch')
        
        return partial_params

    def forward(self, sent_embeddings, z):

        output_64, output_128, output_256, mean, logsigma = self.generator_256(sent_embeddings, z)
        scale_256 = self.generator_256.keep_out_256.detach() 
        scale_512 = self.scale_512(scale_256)
        up_img_256 = F.upsample(output_256.detach(), (512,512), mode='bilinear')

        output_512 = self.tensor_to_img_512(scale_512)

        # self-regularize the 512 generator
        pwloss =  F.l1_loss(output_512, up_img_256)

        return output_64, output_128, output_256, output_512, pwloss

#-----Define Inceptionv3 model for Generator percetual loss------------
# class Inceptionv3(nn.Module):
#     def __init__(self):
#         super(Inceptionv3, self).__init__()
#         model = models.inception_v3()
#         url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
#         model.load_state_dict(model_zoo.load_url(url))
#         for param in model.parameters():
#             param.requires_grad = False
#         print('Load pretrained model from ', url)
#         # print(model)
#
#         self.define_module(model)
#
#     def define_module(self, model):
#         self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
#         self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
#         self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
#         self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
#         self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
#         self.Mixed_5b = model.Mixed_5b
#         self.Mixed_5c = model.Mixed_5c
#         self.Mixed_5d = model.Mixed_5d
#         self.Mixed_6a = model.Mixed_6a
#         self.Mixed_6b = model.Mixed_6b
#         self.Mixed_6c = model.Mixed_6c
#         self.Mixed_6d = model.Mixed_6d
#         self.Mixed_6e = model.Mixed_6e
#         self.Mixed_7a = model.Mixed_7a
#         self.Mixed_7b = model.Mixed_7b
#         self.Mixed_7c = model.Mixed_7c
#
#     def forward(self, x):
#         features = None
#         # --> fixed-size input: batch x 3 x 299 x 299
#         x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
#         # 299 x 299 x 3
#         x = self.Conv2d_1a_3x3(x)
#         # 149 x 149 x 32
#         x = self.Conv2d_2a_3x3(x)
#         # 147 x 147 x 32
#         x = self.Conv2d_2b_3x3(x)
#         # 147 x 147 x 64
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 73 x 73 x 64
#         x = self.Conv2d_3b_1x1(x)
#         # 73 x 73 x 80
#         x = self.Conv2d_4a_3x3(x)
#         # 71 x 71 x 192
#
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
#         # 35 x 35 x 192
#         x = self.Mixed_5b(x)
#         # 35 x 35 x 256
#         x = self.Mixed_5c(x)
#         # 35 x 35 x 288
#         x = self.Mixed_5d(x)
#         # 35 x 35 x 288
#
#         x = self.Mixed_6a(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6b(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6c(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6d(x)
#         # 17 x 17 x 768
#         x = self.Mixed_6e(x)
#         # 17 x 17 x 768
#
#         # image region features
#         features = x
#         # 17 x 17 x 768
#
#         x = self.Mixed_7a(x)
#         # 8 x 8 x 1280
#         x = self.Mixed_7b(x)
#         # 8 x 8 x 2048
#         x = self.Mixed_7c(x)
#         # 8 x 8 x 2048
#         feature = x
#         x = F.avg_pool2d(x, kernel_size=8)
#         # 1 x 1 x 2048
#         # x = F.dropout(x, training=self.training)
#         # 1 x 1 x 2048
#         x = x.view(x.size(0), -1)
#         # 2048
#         # global image features
#         return feature


#-----Define Vgg16 model for Generator percetual loss------------
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        model = models.vgg16(pretrained=True)
        # url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        features = model.features
        # print('Load pretrained model from ', url)
        print('load done pretrained Vgg16')
        print('exact feature from Vgg16 relu2-2')

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])


    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        # h = self.to_relu_3_3(h)
        # h_relu_3_3 = h
        # h = self.to_relu_4_3(h)
        # h_relu_4_3 = h
        out = h_relu_2_2
        # out = h_relu_2_2
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out