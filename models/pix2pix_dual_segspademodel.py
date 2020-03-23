"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from torch import nn


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE,self.netD_fine = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionSeg = nn.NLLLoss2d(ignore_index=255)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image, fine_label, input_semantics_fine = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, fine_label,input_semantics_fine)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, input_semantics_fine)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, seg_out, fake_fine_image, seg_fcn_out= self.generate_fake(input_semantics, real_image, input_semantics_fine)
                seg_fcn_out = torch.argmax(seg_out,dim=1).unsqueeze(0)
                # seg_out = seg_out.float()
                # seg_out = (seg_out/255.0 * 2.0) - 1
                # print(seg_out.size())
                seg_label = self.FloatTensor(1, 35, 256, 512).zero_()
                seg_label = seg_label.scatter_(1, seg_fcn_out, 1.0)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
            # netD_fine optimizer
            D_params += list(self.netD_fine.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netD_fine, 'D_fine', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        # netD = networks.define_D(opt) if opt.isTrain else None
        if opt.isTrain:
            opt.label_nc = opt.label_nc-1
            netD = networks.define_D(opt)
        else:
            netD_fine = None

        netE = networks.define_E(opt) if opt.use_vae else None
        if opt.isTrain:
            opt.label_nc = (opt.label_nc+1)
            netD_fine = networks.define_D(opt)
        else:
            netD_fine = None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                netD_fine = util.load_network(netD_fine, 'D', opt.which_epoch, opt)
            else:
                netD = None
                netD_fine = None
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE, netD_fine

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):

        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            data['fine_label'] = data['fine_label'].long().cuda()
        if self.opt.box:
            input_semantics = data['label'].float()

            label_map = data['fine_label']
            label_map = label_map.view(label_map.size(0),1,label_map.size(1),label_map.size(2))
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics_fine = input_label.scatter_(1, label_map, 1.0)
        else:
            # create one-hot label map
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            if self.opt.box:
                input_semantics_fine = torch.cat((input_semantics_fine, instance_edge_map), dim=1)
            else:
                input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image'], data['fine_label'], input_semantics_fine

    def compute_generator_loss(self, input_semantics, real_image, fine_label,input_semantics_fine):
        G_losses = {}

        fake_image, KLD_loss, seg_out, fake_fine_image,fcn_box_out = self.generate_fake(
            input_semantics, real_image, input_semantics_fine,compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        pred_fake_fine, pred_real_fine = self.discriminate_fine(
            input_semantics_fine, fake_fine_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)
        G_losses['GAN_fine'] = self.criterionGAN(pred_fake_fine, True,
                                            for_discriminator=False)
        if self.opt.seg:
            G_losses['segfine'] = self.criterionSeg(seg_out,fine_label)
        else:
            seg_out.detach()
        if self.opt.dual_segspade:
            G_losses['seg_spade'] = self.criterionSeg(fcn_box_out,fine_label)
        else:
            fcn_box_out.detach()


        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, input_semantics_fine):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _, fake_fine_image, _ = self.generate_fake(input_semantics, real_image,input_semantics_fine)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            fake_fine_image = fake_fine_image.detach()
            fake_fine_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)
        pred_fake_fine, pred_real_fine = self.discriminate_fine(
            input_semantics_fine, fake_fine_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        D_losses['D_Fake_fine'] = self.criterionGAN(pred_fake_fine, False,
                                               for_discriminator=True)
        D_losses['D_real_fine'] = self.criterionGAN(pred_real_fine, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, input_semantics_fine,compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
        if self.opt.dual:
            fake_image,fake_fine_image,seg_out = self.netG(input_semantics,input_semantics_fine, z=z)

            assert (not compute_kld_loss) or self.opt.use_vae, \
                "You cannot compute KLD loss if opt.use_vae == False"

            return fake_image, KLD_loss, seg_out, fake_fine_image
        elif self.opt.dual_segspade:

            fake_image,fake_fine_image,seg_out,fcn_box_out = self.netG(input_semantics,input_semantics_fine, z=z)

            assert (not compute_kld_loss) or self.opt.use_vae, \
                "You cannot compute KLD loss if opt.use_vae == False"

            return fake_image, KLD_loss, seg_out, fake_fine_image, fcn_box_out
        else:
            fake_image,seg_out = self.netG(input_semantics, z=z)

            assert (not compute_kld_loss) or self.opt.use_vae, \
                "You cannot compute KLD loss if opt.use_vae == False"

            return fake_image, KLD_loss, seg_out

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def discriminate_fine(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD_fine(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
