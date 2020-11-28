#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: sa.py
@time: 2020/9/19 11:56
@version 1.0
@desc:
"""
import os

import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.modules.activation import LeakyReLU, ReLU
from torch.nn.modules.container import ModuleList

from networks import *
from utils import get_scheduler, weights_init, load_vgg16, vgg_preprocess, get_model_list


class MultiScaleContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm, activation=activ, pad_type=pad_type)]
        self.n_downsample = n_downsample
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm,
                                 activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        x = self.model[0](x)

        codes = []
        for m in self.model[1:1+self.n_downsample]:
            x = m(x)
            codes.append(x)

        x = self.model[-1](x)

        codes[-1] = x

        return x, codes


class MultiScaleStyleEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm, activation=activ, pad_type=pad_type)]
        self.n_downsample = n_downsample
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm,
                                 activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        x = self.model[0](x)
        codes = []
        for m in self.model[1:1+self.n_downsample]:
            x = m(x)
            codes.append(x)
        x = self.model[-1](x)

        codes[-1] = x
        return x, codes


class MultiScaleSADecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero', scale=1):
        super(Decoder, self).__init__()
        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]

        self.sanet_head = SANet(dim, dim, dim)

        self.sanet_modules = ModuleList(
            [SANet(dim // (2 ** l), dim // (2 ** l), dim // (2 ** l)) for l in range(scale)])

        self.n_unpsample = n_upsample
        self.level = scale

        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3,
                                   norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, content_feats, style_feats):

        x = self.sanet_head(content_feats[-1], style_feats[-1])

        x = self.model[0](x)

        for i in range(self.n_unpsample):
            if i < self.level:
                x = self.sanet_modules[i](
                    x, style_feats[-(i+1)])
            x = self.model[1+i](x)

        x = self.model[-1](x)

        return x


class MultiScaleSAGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(SAGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        self.mlp_dim = params['mlp_dim']
        self.style_dim = params['style_dim']
        self.style_encoder_type = params['style_encoder_type']
        self.style_code_dim = params['style_code_dim']
        self.n_blk = params['n_blk']
        self.activ = params['activ']
        self.mlp_type = params['mlp_type']
        self.level = params['level']
        self.mapping_layers = params['mapping_layers']
        self.use_mapping = params['use_mapping']

        # content encoder
        self.enc_content = MultiScaleContentEncoder(
            n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        self.enc_style = MultiScaleStyleEncoder(
            n_downsample, n_res, input_dim, dim, 'none', activ, pad_type=pad_type)

        self.mapping_nets = []
        map_dim = self.enc_content.output_dim
        for i in range(self.mapping_layers):
            self.mapping_nets.append(
                Conv2dBlock(map_dim, map_dim, 3, 1, 1,
                            activation=activ, pad_type=pad_type),
                Conv2dBlock(map_dim, map_dim, 1),
                LeakyReLU(inplace=True)
            )
        self.mapping_nets = nn.Sequential(*self.mapping_nets)
        self.dec = MultiScaleSADecoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='none',
                                       activ=activ,
                                       pad_type=pad_type, scale=params['scale'])

    def forward(self, images):
        content, style_fake, _ = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        content_code, content_feats = self.enc_content(images)
        style_code, style_feats = self.enc_style(images)
        return content_code, content_feats, style_code, style_feats

    def decode(self, content_feats, style_feats, enable_mapping=False):
        if enable_mapping:
            style_feats[-1] = self.mapping_nets(
                style_feats[-1]).view_as(content_feats[-1])
        images = self.dec(content_feats, style_feats)
        return images


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), GramMatrix()(target))
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return nn.MSELoss(input_mean, target_mean) + \
        nn.MSELoss(input_std, target_std)


class MultiScaleSANET_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MultiScaleSANET_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = None
        self.gen_b = None
        self.gen_a = MultiScaleSAGen(
            hyperparameters['input_dim_a'], hyperparameters['gen'])
        # auto-encoder for domain b
        self.gen_b = MultiScaleSAGen(
            hyperparameters['input_dim_b'], hyperparameters['gen'])
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'],
                                hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'],
                                hyperparameters['dis'])  # discriminator for domain b
        self.cos_sim = nn.CosineSimilarity()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.content_output_dim = self.gen_a.enc_content.output_dim
        self.style_dim = hyperparameters['gen']['style_dim']
        self.style_encoder_type = hyperparameters['gen']['style_encoder_type']
        self.style_criterion = hyperparameters['gen']['style_criterion']
        self.display_size = hyperparameters['display_size']
        self.scale = hyperparameters['scale']
        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.gram_mse_loss = GramMSELoss()
        self.enable_mapping = hyperparameters['enable_mapping']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + \
            list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(
                hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def cal_cos_distance(self, x1, x2):
        return 1 - self.cos_sim(x1, x2)

    def anti_collapse_criterion(self, s1, s2, x1, x2, eps=1e-12):
        return self.cal_cos_distance(s1.reshape(s1.size(0), -1), s2.reshape(s2.size(0), -1)) / (self.cal_cos_distance(
            x1.reshape(x1.size(0), -1),
            x2.reshape(x2.size(0), -1)) + eps)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def style_recon_criterion(self, input, target):
        if self.style_criterion == 'gram':
            return self.gram_mse_loss(input, target)
        elif self.style_encoder_type == 'mu':
            return calc_style_loss(input, target)
        else:
            return self.recon_criterion(input, target)

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())

        # encode
        c_a, c_a_feats, s_a_prime, s_a_feats = self.gen_a.encode(x_a)
        c_b, c_b_feats, s_b_prime, s_b_feats = self.gen_b.encode(x_b)

        # c_b, s_b_prime, b_feats = self.gen_b.encode(x_b)

        s_a_rn_feats, s_b_rn_feats = self.sample_multi_scale_style_code(
            c_a_feats, c_b_feats)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a_feats, s_a_feats)
        x_b_recon = self.gen_b.decode(c_b_feats, s_b_feats)

        # decode (cross domain) using style code from normal distribution
        x_ba = self.gen_a.decode(
            c_b_feats, s_a_rn_feats, enable_mapping=self.enable_mapping)
        x_ab = self.gen_b.decode(
            c_a_feats, s_b_rn_feats, enable_mapping=self.enable_mapping)

        # decode (cross domain) using style code from style encoder
        x_real_ba = self.gen_a.decode(c_b_feats, s_a_feats)
        x_real_ab = self.gen_b.decode(c_a_feats, s_b_feats)

        # encode again
        c_b_recon, c_b_recon_feats, s_a_recon, s_ba_recon_feats = self.gen_a.encode(
            x_ba)
        c_a_recon, c_a_recon_feats, s_b_recon, s_ab_recon_feats = self.gen_b.encode(
            x_ab)

        c_real_b_recon, c_real_b_recon_feats, s_real_a_recon, s_real_a_recon_feats = self.gen_a.encode(
            x_real_ba)
        c_real_a_recon, c_real_a_recon_feats, s_real_b_recon, s_real_b_recon_feats = self.gen_b.encode(
            x_real_ab)

        # decode again (if needed)
        x_aba = self.gen_a.decode(
            c_a_recon_feats, s_a_feats) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(
            c_b_recon_feats,  s_b_feats) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # image reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        # diversity sensitive loss
        # s_a2, s_b2, a_srn_feats2, b_srn_feats2 = self.sample_style_code(x_a, x_b, c_a, c_b, a_feats, b_feats)
        # x_ba2 = self.gen_a.decode(c_b, s_a2, a_srn_feats2)
        # x_ab2 = self.gen_b.decode(c_a, s_b2, b_srn_feats2)
        # self.loss_diversity_loss_ba = - torch.mean(torch.abs(x_ba2 - x_ba))
        # self.loss_diversity_loss_ab = - torch.mean(torch.abs(x_ab2 - x_ab))

        # anti collapse loss
        s_arn_feats2, s_b_rn_feats2 = self.sample_multi_scale_style_code(
            c_a_feats, c_b_feats)
        x_ba2 = self.gen_a.decode(
            c_b_feats[-1], s_a_rn_feats[-1], s_arn_feats2[-1])
        x_ab2 = self.gen_b.decode(
            c_a_feats[-1], s_b_rn_feats[-1], s_b_rn_feats[-1])
        # self.loss_diversity_loss_ba = - torch.mean(torch.abs(x_ba2 - x_ba))
        # self.loss_diversity_loss_ab = - torch.mean(torch.abs(x_ab2 - x_ab))
        self.loss_anti_collapse_ba = self.anti_collapse_criterion(
            s_a_rn_feats[-1], s_arn_feats2[-1], x_ba, x_ba2).mean()
        self.loss_anti_collapse_ab = self.anti_collapse_criterion(
            s_b_rn_feats[-1], s_b_rn_feats2[-1], x_ab, x_ab2).mean()

        # latent reconstruction loss(style encoder branch)
        for i in range(self.scale):
            self.loss_gen_recon_real_s_a += self.style_recon_criterion(
                s_real_a_recon_feats[i], s_a_feats[i])
            self.loss_gen_recon_real_s_b += self.style_recon_criterion(
                s_real_b_recon_feats[i], s_b_feats[i])
            self.loss_gen_recon_real_c_a += self.recon_criterion(
                c_real_a_recon_feats[i], c_a_feats[i])
            self.loss_gen_recon_real_c_b += self.recon_criterion(
                c_real_b_recon_feats[i], c_b_feats[i])

        # latent reconstruction loss(random sample branch)
        for i in range(self.scale):
            self.loss_gen_recon_s_a += self.style_recon_criterion(
                s_ba_recon_feats[i], s_a_rn_feats[i])
            self.loss_gen_recon_s_b += self.style_recon_criterion(
                s_ab_recon_feats[i], s_b_rn_feats[i])
            self.loss_gen_recon_c_a += self.recon_criterion(
                c_a_recon_feats[i], c_a_feats[i])
            self.loss_gen_recon_c_b += self.recon_criterion(
                c_b_recon_feats[i], c_b_feats[i])

        self.loss_gen_cycrecon_x_a = self.recon_criterion(
            x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(
            x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        self.loss_gen_adv_real_a = self.dis_a.calc_gen_loss(x_real_ba)
        self.loss_gen_adv_real_b = self.dis_b.calc_gen_loss(x_real_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(
            self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(
            self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
            hyperparameters['gan_w'] * self.loss_gen_adv_real_a + \
            hyperparameters['gan_w'] * self.loss_gen_adv_real_b + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_real_s_a + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_real_c_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_real_s_b + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_real_c_b + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
            hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
            hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
            hyperparameters['ds_w'] * self.loss_anti_collapse_ab + \
            hyperparameters['ds_w'] * self.loss_anti_collapse_ba

        # loss_dict = {'gen_adv_a': hyperparameters['gan_w'] * self.loss_gen_adv_a.item(),
        #              'gan_adv_b': hyperparameters['gan_w'] * self.loss_gen_adv_b.item(),
        #              'gen_recon_x_a': hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a.item(),
        #              'gen_recon_s_a': hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a.item(),
        #              'gen_recon_c_a': hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a.item(),
        #              'gen_recon_x_b': hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b.item(),
        #              'gen_recon_s_b': hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b.item(),
        #              'gen_recon_c_b': hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b.item(),
        #              'gen_total_loss': self.loss_gen_total.item()
        #              }

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample_multi_scale_style_code(self, a_feats, b_feats):
        s_a_feats = []
        s_b_feats = []
        for i in range(self.scale):
            s_a_feat = Variable(torch.randn_like(a_feats[i]).cuda())
            s_a_feats.append(s_a_feat)
            s_b_feat = Variable(torch.rand_like(b_feats[i]).cuda())
            s_b_feats.append(s_b_feat)

        return s_a_feats, s_b_feats

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample_ref(self, x_a, x_b, x_a_ref, x_b_ref):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, c_a_feats, s_a, s_a_feats = self.gen_a.encode(
                x_a[i].unsqueeze(0))
            c_b, c_b_feats, s_b, s_b_feats = self.gen_b.encode(
                x_b[i].unsqueeze(0))
            c_a_ref, c_a_ref_feats, s_a_ref, s_a_ref_feats = self.gen_a.encode(
                x_a_ref[i].unsqueeze(0))
            c_b_ref, c_b_ref_feats, s_b_ref, s_b_ref_feats = self.gen_b.encode(
                x_b_ref[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a_feats, s_a_feats))
            x_b_recon.append(self.gen_b.decode(c_b_feats, s_a_feats))
            x_ba1.append(self.gen_a.decode(c_b_feats, s_a_ref_feats))
            # x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a_feats, s_b_ref_feats))
            # x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        #
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        # x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        # x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        x_ba1 = torch.cat(x_ba1)
        x_ab1 = torch.cat(x_ab1)
        self.train()
        return x_a, x_a_recon, x_b_ref, x_ab1, x_b, x_b_recon, x_a_ref, x_ba1

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, c_a_feats, s_a, s_a_feats = self.gen_a.encode(x_a)
        c_b, c_b_feats, s_b, s_b_feats = self.gen_b.encode(x_b)

        s_a_rn_feats, s_b_rn_feats = self.sample_multi_scale_style_code(
            c_a_feats, c_b_feats)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b_feats, s_a_rn_feats)
        x_ab = self.gen_b.decode(c_a_feats, s_b_rn_feats)

        x_real_ba = self.gen_a.decode(c_b_feats, s_a_feats)
        x_real_ab = self.gen_b.decode(c_a_feats, s_b_feats)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        self.loss_dis_real_a = self.dis_a.calc_dis_loss(
            x_real_ba.detach(), x_a)
        self.loss_dis_real_b = self.dis_b.calc_dis_loss(
            x_real_ab.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b + \
            hyperparameters['gan_w'] * self.loss_dis_real_a + hyperparameters[
            'gan_w'] * self.loss_dis_real_b

        # loss_dict = {
        #     'dis_a': self.loss_dis_a.item(),
        #     'dis_b': self.loss_dis_b.item(),
        #     'dis_total': self.loss_dis_total.item()
        # }
        self.loss_dis_total.backward()
        self.dis_opt.step()
        # return loss_dict

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(),
                    'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(),
                    'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)
