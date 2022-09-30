# -*- coding: UTF-8 -*-
"""
@Function:
@File: pce_backbone.py
@Date: 2022/6/2 16:41 
@Author: Hever
"""
import torch
import torch.nn as nn
import functools


class OriUnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(OriUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        conv_d1 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        relu_d1 = nn.LeakyReLU(0.2, inplace=True)
        norm_d1 = norm_layer(inner_nc)
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)

        conv_u1 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        relu_u1 = nn.ReLU(inplace=True)
        norm_u1 = norm_layer(outer_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, conv_d1, relu_d1, norm_d1]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, relu_d1, conv_d1, norm_d1]
            up = [uprelu, upconv, upnorm, relu_u1, conv_u1, norm_u1]

        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)


class PCEBackbone(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, need_feature=False):
        super(PCEBackbone, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.in_conv1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                      norm_layer(ngf))
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, kernel_size=1),
            nn.Tanh())

        unet_block4 = OriUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=ngf * 9,
                                                 norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        # unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block3 = OriUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=ngf * 5, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block2 = OriUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=ngf * 3, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block1 = OriUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=ngf * 1, norm_layer=norm_layer, use_dropout=use_dropout) # add the outermost layer
        self.down1, self.up1 = unet_block1.down, unet_block1.up
        self.down2, self.up2 = unet_block2.down, unet_block2.up
        self.down3, self.up3 = unet_block3.down, unet_block3.up
        self.down4, self.up4 = unet_block4.down, unet_block4.up
        # self.down5, self.up5 = unet_block5.down, unet_block5.up
        self.h_conv1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))
        self.h_conv2 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))
        self.h_conv3 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                                     norm_layer(ngf))

        # self.input_low = input_low
        self.need_feature = need_feature
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.inner_conv = nn.Sequential(nn.Conv2d(ngf * 8 + 3, ngf * 8, kernel_size=3, padding=1, bias=use_bias),
                                 norm_layer(ngf))

    def forward(self, input_list, need_feature=False):
        """Standard forward"""
        x_high, down_h1, down_h2, down_h3, down4 = input_list
        down_h1_feature = self.h_conv1(down_h1)  # 128
        down_h2_feature = self.h_conv2(down_h2)  # 64
        down_h3_feature = self.h_conv3(down_h3)  # 32

        in1 = self.in_conv1(x_high)
        # downsample
        d1 = self.down1(in1)

        d2 = self.down2(torch.cat([d1, down_h1_feature], dim=1))
        d3 = self.down3(torch.cat([d2, down_h2_feature], dim=1))
        d4 = self.down4(torch.cat([d3, down_h3_feature], dim=1))

        d4 = self.leaky_relu(d4)
        d4 = self.inner_conv(torch.cat([d4, down4], 1))

        # upsample
        u4 = self.up4(d4)
        # u4 = self.up4(torch.cat([u5, d4], 1))
        u3 = self.up3(torch.cat([u4, d3], 1))
        u2 = self.up2(torch.cat([u3, d2], 1))
        u1 = self.up1(torch.cat([u2, d1], 1))
        out1 = self.out_conv(torch.cat([u1, in1], 1))
        if need_feature:
            under_activate_features = [in1, d1, d2, d3, d4]
            return out1, under_activate_features
        return out1