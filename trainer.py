"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from tensorboardX import SummaryWriter

from networks import AdaINGen, MsImageDis, VAEGen, SAGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
from pathlib import Path
import tensorboardX
import torch
import torch.nn as nn
import os


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = None
        self.gen_b = None
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

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
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        x_real_ba = self.gen_a.decode(c_b, s_a_prime)
        x_real_ab = self.gen_a.decode(c_a, s_b_prime)

        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        c_real_b_recon, s_real_a_recon = self.gen_a.encode(x_real_ba)
        c_real_a_recon, s_real_b_recon = self.gen_a.encode(x_real_ba)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_recon_real_s_a = self.recon_criterion(s_real_a_recon, s_a_prime)
        self.loss_gen_recon_real_s_b = self.recon_criterion(s_real_b_recon, s_b_prime)
        self.loss_gen_recon_real_c_a = self.recon_criterion(c_real_a_recon, c_a)
        self.loss_gen_recon_real_c_b = self.recon_criterion(c_real_b_recon, c_b)

        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
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
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

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
        # return loss_dict

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # s_a2 = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        # s_b2 = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        # s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        # encode
        # c_a, _ = self.gen_a.encode(x_a)
        # c_b, _ = self.gen_b.encode(x_b)

        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)

        # c_a, s_a_prime = self.gen_a.encode(x_a)
        # c_b, _ = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        x_real_ba = self.gen_a.decode(c_b, s_a_prime)
        x_real_ab = self.gen_b.decode(c_a, s_b_prime)

        self.loss_dis_real_a = self.dis_a.calc_dis_loss(x_real_ba.detach(), x_a)
        self.loss_dis_real_b = self.dis_b.calc_dis_loss(x_real_ab.detach(), x_b)
        # D loss

        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b + \
                              hyperparameters['gan_w'] * self.loss_dis_real_a + hyperparameters['gan_w'] * self.loss_dis_real_b

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
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

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
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class SANET_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(SANET_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = None
        self.gen_b = None
        if hyperparameters['gen_type'] == 'sanet':
            self.gen_a = SAGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
            self.gen_b = SAGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        else:
            self.gen_a = AdaINGen(hyperparameters['input_dim_a'],
                                  hyperparameters['gen'])  # auto-encoder for domain a
            self.gen_b = AdaINGen(hyperparameters['input_dim_b'],
                                  hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'],
                                hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'],
                                hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.content_output_dim = self.gen_a.enc_content.output_dim
        self.style_dim = hyperparameters['gen']['style_dim']
        self.style_encoder_type = hyperparameters['gen']['style_encoder_type']
        self.display_size = hyperparameters['display_size']
        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        if self.style_encoder_type == 'mirror':
            self.s_a = torch.randn(display_size, self.content_output_dim, 64, 64).cuda()
            self.s_b = torch.randn(display_size, self.content_output_dim, 64, 64).cuda()
        elif self.style_encoder_type == 'mapping':
            # generate from mapping network after proper training
            self.s_a = None
            self.s_b = None
        elif self.style_encoder_type == 'multi-level':
            self.s_a = None
            self.s_b = None
            self.s_a_feats = None
            self.s_b_feats = None

            # self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        # self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

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
        c_a, s_a_prime, a_feats = self.gen_a.encode(x_a)
        c_b, s_b_prime, b_feats = self.gen_b.encode(x_b)

        s_a, s_b, a_srn_feats, b_srn_feats = self.sample_style_code(x_a, x_b, c_a, c_b, a_feats, b_feats)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime, a_feats)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime, b_feats)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a, a_srn_feats)
        x_ab = self.gen_b.decode(c_a, s_b, b_srn_feats)
        # encode again
        c_b_recon, s_a_recon, ba_feats = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon, ab_feats = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime, a_feats) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime, b_feats) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

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

    def sample_style_code(self, x_a, x_b, c_a, c_b, a_feats, b_feats):
        if self.style_encoder_type == 'mirror':
            s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        elif self.style_encoder_type == 'mapping':
            s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
            assert self.gen_a.mlp is not None
            assert self.gen_b.mlp is not None
            s_a = self.gen_a.mlp(s_a)
            s_b = self.gen_b.mlp(s_b)
            # a mapping style code must be reshaped as content code
            s_a = s_a.view_as(c_a)
            s_b = s_b.view_as(c_b)
        elif self.style_encoder_type == 'multi-level':
            return self.sample_multi_level_style_code(x_a, x_b, a_feats, b_feats)

        return s_a, s_b, None, None

    def sample_multi_level_style_code(self, x_a, x_b, a_feats, b_feats, init_s_a_feats=None, init_s_b_feats=None,
                                      use_map=True):
        s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        s_a_feats = []
        s_b_feats = []
        a_mapping_nets = self.gen_a.enc_style.mapping_nets
        b_mapping_nets = self.gen_b.enc_style.mapping_nets
        for idx, am in enumerate(a_mapping_nets):
            # sample style code from uniform distribution
            if init_s_a_feats is None:
                s_a_feat = Variable(torch.randn_like(a_feats[idx]).cuda())
            else:
                s_a_feat = init_s_a_feats[idx]
            # project to the other common space using mapping network
            if use_map:
                s_a_feats.append(am(s_a_feat))
            else:
                s_a_feats.append(s_a_feat)
        for idx, bm in enumerate(b_mapping_nets):
            if init_s_b_feats is None:
                s_b_feat = Variable(torch.randn_like(b_feats[idx]).cuda())
            else:
                s_b_feat = init_s_b_feats[idx]
            if use_map:
                s_b_feats.append(bm(s_b_feat))
            else:
                s_b_feats.append(s_b_feat)
        # s_a_feats = torch.cat(s_a_feats).permute(1, 0, 2, 3, 4).unsqueeze(2)
        # s_b_feats = torch.cat(s_b_feats).permute(1, 0, 2, 3, 4).unsqueeze(2)
        return s_a, s_b, s_a_feats, s_b_feats

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        c_a, s_a_fake, a_feats = self.gen_a.encode(x_a)
        c_b, s_b_fake, b_feats = self.gen_b.encode(x_b)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        if self.style_encoder_type == 'mirror':
            # init style code of A domain and style code of B domain
            # if self.s_a is None or self.s_b is None:
            # self.s_a = torch.randn(self.display_size, self.style_dim, 1, 1).cuda()
            # self.s_b = torch.randn(self.display_size, self.style_dim, 1, 1).cuda()
            self.s_a = torch.randn(self.display_size, self.content_output_dim, 64, 64).cuda()
            self.s_b = torch.randn(self.display_size, self.content_output_dim, 64, 64).cuda()
            s_a1 = Variable(self.s_a)
            s_b1 = Variable(self.s_b)
            # s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            # s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
            # s_a2 = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
            # s_b2 = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
            s_a2, s_b2, a2_srn_feats, b2_srn_feats = self.sample_style_code(x_a, x_b, c_a, c_b, a_feats, b_feats)
            for i in range(x_a.size(0)):
                c_a, s_a_fake, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        elif self.style_encoder_type == 'multi-level':
            if self.s_a_feats is None or self.s_b_feats is None:
                # init style code only once of each level, and fixed as input of mapping network
                self.s_a, self.s_b, self.s_a_feats, self.s_b_feats = self.sample_multi_level_style_code(x_a, x_b,
                                                                                                        a_feats,
                                                                                                        b_feats,
                                                                                                        use_map=False)
            # sample multi-level style codes by input fixed noises mapping network at each val stage
            _, _, a1_srn_feats, b1_srn_feats = self.sample_multi_level_style_code(x_a, x_b, a_feats, b_feats,
                                                                                  init_s_a_feats=self.s_a_feats,
                                                                                  init_s_b_feats=self.s_b_feats,
                                                                                  use_map=True)
            s_a2, s_b2, a2_srn_feats, b2_srn_feats = self.sample_style_code(x_a, x_b, c_a, c_b, a_feats, b_feats)
            s_a1 = Variable(self.s_a)
            s_b1 = Variable(self.s_b)
            a1_srn_feats = torch.stack(a1_srn_feats, dim=1).unsqueeze(2)
            b1_srn_feats = torch.stack(b1_srn_feats, dim=1).unsqueeze(2)
            a2_srn_feats = torch.stack(a2_srn_feats, dim=1).unsqueeze(2)
            b2_srn_feats = torch.stack(b2_srn_feats, dim=1).unsqueeze(2)
            for i in range(x_a.size(0)):
                c_a, s_a_fake, a_feats = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake, b_feats = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake, a_feats))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake, b_feats))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0), a1_srn_feats[i]))
                x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0), a2_srn_feats[i]))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0), b1_srn_feats[i]))
                x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0), b2_srn_feats[i]))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def sample_ref(self, x_a, x_b, x_a_ref, x_b_ref):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            c_ref_a, s_ref_a, _ = self.gen_a.encode(x_a_ref[i].unsqueeze(0))
            c_ref_b, s_ref_b, _ = self.gen_b.encode(x_b_ref[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_ref_a))
            # x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_ref_b))
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
        s_a = Variable(torch.randn(x_a.size(0), self.content_output_dim, 64, 64).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.content_output_dim, 64, 64).cuda())
        # encode
        c_a, _, a_feats = self.gen_a.encode(x_a)
        c_b, _, b_feats = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a, a_feats)
        x_ab = self.gen_b.decode(c_a, s_b, b_feats)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b

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
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
