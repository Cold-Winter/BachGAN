"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

from models.networks.architecture import DualSPADEResnetBlock as DualSPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
            if self.opt.retrival_memory:
                self.fc = nn.Conv2d(self.opt.semantic_nc * 1, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.softmax = nn.LogSoftmax()
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_seg = nn.Conv2d(final_nc, self.opt.semantic_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


        # fcn encoder
        nhidden = 128
        self.encode_shared = nn.Sequential(
            nn.Conv2d(self.opt.semantic_nc, self.opt.semantic_nc, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.memory_shared = nn.Sequential(
            nn.Conv2d(self.opt.semantic_nc, self.opt.semantic_nc, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_1 = nn.Conv2d(35, 64, 3, padding=1)
        self.conv1_1_memory = nn.Conv2d(70, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 5, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(4, stride=4, ceil_mode=True)  # 1/2

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 5, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(4, stride=4, ceil_mode=True)  # 1/4
        self.upscore = nn.ConvTranspose2d(128, 35, 32, stride=16,
                                          bias=False)

        self.drop2 = nn.Dropout2d()

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, retrival_label_list, z=None):
        seg = input

        seg_and_retrivallabels = torch.cat([seg,retrival_label_list],dim=1)
        retrival_label_list_batchshape = seg_and_retrivallabels.view([-1,self.opt.semantic_nc]+ [seg_and_retrivallabels.size(-2),seg_and_retrivallabels.size(-1)])
        actv = self.encode_shared(retrival_label_list_batchshape)
        actv = actv.view([-1,4] + [actv.size(-3),actv.size(-2),actv.size(-1)])


        m_0 = actv[:,0,:,:,:] + (actv[:,1,:,:,:] + actv[:,2,:,:,:] + actv[:,3,:,:,:])/3
        m_1 = self.memory_shared(m_0)


        # pool_box_out = retrival_label_list_batchshape.view([-1,3,35] + [retrival_label_list.size(-2),retrival_label_list.size(-1)])
        # c_out_test = (pool_box_out[:,0,:,:,:] + pool_box_out[:,1,:,:,:] + pool_box_out[:,2,:,:,:])/3



        # h = retrival_label_list_batchshape
        # h = self.relu1_1(self.conv1_1(h))
        # h = self.relu1_2(self.conv1_2(h))
        # h = self.pool1(h)

        # h = self.relu2_1(self.conv2_1(h))
        # h = self.relu2_2(self.conv2_2(h))
        # h = self.pool2(h)
        # h = self.drop2(h)
        # h = self.upscore(h)
        # fcn_box_out = F.interpolate(h, size=(256, 512))
        # fcn_box_out = fcn_box_out.view([-1,3,35] + [retrival_label_list.size(-2),retrival_label_list.size(-1)])
        # c_out = (fcn_box_out[:,0,:,:,:] + fcn_box_out[:,1,:,:,:] + fcn_box_out[:,2,:,:,:])/3
        
        # m_0 = input

        # m_and_c = torch.cat([m_0,c_out],dim=1)

        # h = self.relu1_1(self.conv1_1_memory(m_and_c))
        # h = self.relu1_2(self.conv1_2(h))
        # h = self.pool1(h)

        # h = self.relu2_1(self.conv2_1(h))
        # h = self.relu2_2(self.conv2_2(h))
        # h = self.pool2(h)
        # h = self.drop2(h)
        # h = self.upscore(h)
        # m_1 = F.interpolate(h, size=(256, 512))


        # h = retrival_label_list_batchshape
        # h = self.relu1_1(self.conv1_1(h))
        # h = self.relu1_2(self.conv1_2(h))
        # h = self.pool1(h)

        # h = self.relu2_1(self.conv2_1(h))
        # h = self.relu2_2(self.conv2_2(h))
        # h = self.pool2(h)
        # h = self.drop2(h)
        # h = self.upscore(h)
        # fcn_box_out = F.interpolate(h, size=(256, 512))
        # fcn_box_out = fcn_box_out.view([-1,3,35] + [retrival_label_list.size(-2),retrival_label_list.size(-1)])
        # c_out_1 = (fcn_box_out[:,0,:,:,:] + fcn_box_out[:,1,:,:,:] + fcn_box_out[:,2,:,:,:])/3


        # m_and_c = torch.cat([m_1,c_out_1],dim=1)

        # h = self.relu1_1(self.conv1_1_memory(m_and_c))
        # h = self.relu1_2(self.conv1_2(h))
        # h = self.pool1(h)

        # h = self.relu2_1(self.conv2_1(h))
        # h = self.relu2_2(self.conv2_2(h))
        # h = self.pool2(h)
        # h = self.drop2(h)
        # h = self.upscore(h)
        # m_2 = F.interpolate(h, size=(256, 512))
        # m_2.detach()
        # seg = torch.cat([seg,m_2],dim=1)

        # seg = torch.cat([seg,m_2],dim=1)
        # seg = retrival_label_list[:,0:35,:,:]
        # seg = seg[:,0:35,:,:]
        # print(seg.size())
        # seg = c_out_test

        # seg = torch.cat([seg,retrival_label_list],dim=1)
        seg = m_1 + actv[:,0,:,:,:]


        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            
            if self.opt.retrival_memory:
                x = F.interpolate(seg[:,0:self.opt.semantic_nc,:,:], size=(self.sh, self.sw))
            else:
                x = F.interpolate(seg, size=(self.sh, self.sw))

            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        seg_out = self.conv_seg(F.leaky_relu(x, 2e-1))
        seg_out = self.softmax(seg_out)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x,seg_out


class DUALSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
            self.fc_fine = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = DualSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = DualSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = DualSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = DualSPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = DualSPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.softmax = nn.LogSoftmax()
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_img_fine = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_seg = nn.Conv2d(final_nc, self.opt.label_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, input_fine, z=None):
        seg = input
        seg_fine = input_fine

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
            x_fine = F.interpolate(seg_fine, size=(self.sh, self.sw))
            x_fine = self.fc_fine(x_fine)

        x,x_fine= self.head_0(x,x,seg,seg_fine)

        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.G_middle_0(x,x_fine,seg,seg_fine)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x_fine = self.up(x_fine)

        x,x_fine = self.G_middle_1(x,x_fine,seg,seg_fine)

        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_0(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_1(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_2(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_3(x,x_fine,seg,seg_fine)


        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x_fine = self.up(x_fine)
            x,x_fine = self.up_4(x,x_fine,seg,seg_fine)

        seg_out = self.conv_seg(F.leaky_relu(x, 2e-1))
        seg_out = self.softmax(seg_out)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)


        x_fine = self.conv_img_fine(F.leaky_relu(x_fine, 2e-1))
        x_fine = F.tanh(x_fine)

        return x,x_fine,seg_out

class DUALSEGSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)




        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d((self.opt.semantic_nc-1)*2, 16 * nf, 3, padding=1)
            #if semantic_nc + 1 since the fine lable have instance map --- further experiment need to be done
            self.fc_fine = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        #simple fcn from box to seg
        self.conv1_1 = nn.Conv2d(self.opt.semantic_nc-1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 5, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(4, stride=4, ceil_mode=True)  # 1/2

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 5, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(4, stride=4, ceil_mode=True)  # 1/4

        self.upscore = nn.ConvTranspose2d(128, self.opt.label_nc, 32, stride=16,
                                          bias=False)

        self.drop2 = nn.Dropout2d()

        self.head_0 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = DualSPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = DualSPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = DualSPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = DualSPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = DualSPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = DualSPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        self.softmax = nn.LogSoftmax()
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_img_fine = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.conv_seg = nn.Conv2d(final_nc, self.opt.label_nc, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, input_fine, z=None):
        seg = input
        seg_fine = input_fine

        h = seg
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        h = self.drop2(h)
        h = self.upscore(h)

        fcn_box_out = F.interpolate(h, size=(256, 512))
        fcn_box_out = self.softmax(fcn_box_out)

        seg = torch.cat((seg,fcn_box_out),dim=1)

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
            x_fine = F.interpolate(seg_fine, size=(self.sh, self.sw))
            x_fine = self.fc_fine(x_fine)

        x,x_fine= self.head_0(x,x_fine,seg,seg_fine)

        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.G_middle_0(x,x_fine,seg,seg_fine)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x_fine = self.up(x_fine)

        x,x_fine = self.G_middle_1(x,x_fine,seg,seg_fine)

        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_0(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_1(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_2(x,x_fine,seg,seg_fine)
        x = self.up(x)
        x_fine = self.up(x_fine)
        x,x_fine = self.up_3(x,x_fine,seg,seg_fine)


        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x_fine = self.up(x_fine)
            x,x_fine = self.up_4(x,x_fine,seg,seg_fine)

        seg_out = self.conv_seg(F.leaky_relu(x, 2e-1))
        seg_out = self.softmax(seg_out)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)


        x_fine = self.conv_img_fine(F.leaky_relu(x_fine, 2e-1))
        x_fine = F.tanh(x_fine)


        return x,x_fine,seg_out,fcn_box_out


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)

# class SPADEGenerator(BaseNetwork):
#     @staticmethod
#     def modify_commandline_options(parser, is_train):
#         parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#         parser.add_argument('--num_upsampling_layers',
#                             choices=('normal', 'more', 'most'), default='normal',
#                             help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

#         return parser

#     def __init__(self, opt):
#         super().__init__()
#         self.opt = opt
#         nf = opt.ngf

#         self.sw, self.sh = self.compute_latent_vector_size(opt)

#         if opt.use_vae:
#             # In case of VAE, we will sample from random z vector
#             self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
#         else:
#             # Otherwise, we make the network deterministic by starting with
#             # downsampled segmentation map instead of random z
#             self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

#         self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

#         self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
#         self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

#         self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
#         self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
#         self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
#         self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

#         final_nc = nf

#         if opt.num_upsampling_layers == 'most':
#             self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
#             final_nc = nf // 2
#         self.softmax = nn.LogSoftmax()
#         self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
#         self.conv_seg = nn.Conv2d(final_nc, self.opt.semantic_nc, 3, padding=1)

#         self.up = nn.Upsample(scale_factor=2)

#     def compute_latent_vector_size(self, opt):
#         if opt.num_upsampling_layers == 'normal':
#             num_up_layers = 5
#         elif opt.num_upsampling_layers == 'more':
#             num_up_layers = 6
#         elif opt.num_upsampling_layers == 'most':
#             num_up_layers = 7
#         else:
#             raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
#                              opt.num_upsampling_layers)

#         sw = opt.crop_size // (2**num_up_layers)
#         sh = round(sw / opt.aspect_ratio)

#         return sw, sh

#     def forward(self, input, z=None):
#         seg = input

#         if self.opt.use_vae:
#             # we sample z from unit normal and reshape the tensor
#             if z is None:
#                 z = torch.randn(input.size(0), self.opt.z_dim,
#                                 dtype=torch.float32, device=input.get_device())
#             x = self.fc(z)
#             x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
#         else:
#             # we downsample segmap and run convolution
#             x = F.interpolate(seg, size=(self.sh, self.sw))
#             x = self.fc(x)

#         x = self.head_0(x, seg)

#         x = self.up(x)
#         x = self.G_middle_0(x, seg)

#         if self.opt.num_upsampling_layers == 'more' or \
#            self.opt.num_upsampling_layers == 'most':
#             x = self.up(x)

#         x = self.G_middle_1(x, seg)

#         x = self.up(x)
#         x = self.up_0(x, seg)
#         x = self.up(x)
#         x = self.up_1(x, seg)
#         x = self.up(x)
#         x = self.up_2(x, seg)
#         x = self.up(x)
#         x = self.up_3(x, seg)

#         if self.opt.num_upsampling_layers == 'most':
#             x = self.up(x)
#             x = self.up_4(x, seg)
#         seg_out = self.conv_seg(F.leaky_relu(x, 2e-1))
#         seg_out = self.softmax(seg_out)

#         x = self.conv_img(F.leaky_relu(x, 2e-1))
#         x = F.tanh(x)

#         return x,seg_out