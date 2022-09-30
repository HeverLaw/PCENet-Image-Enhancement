# -*- coding: UTF-8 -*-
"""
@Function:
@File: unet_generator.py
@Date: 2021/12/9 19:16 
@Author: Hever
"""
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
# from torchvision.models import resnet18

class Unet_generator_5layers(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet_generator_5layers, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # TODO:cbam模块是在后头，修改位置
        unet_block5 = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)
        unet_block4 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block3 = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block2 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block1 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, outermost=True, norm_layer=norm_layer,
                                              use_dropout=use_dropout)  # add the outermost layer
        self.down1, self.up1 = unet_block1.down, unet_block1.up
        self.down2, self.up2 = unet_block2.down, unet_block2.up
        self.down3, self.up3 = unet_block3.down, unet_block3.up
        self.down4, self.up4 = unet_block4.down, unet_block4.up
        self.down5, self.up5 = unet_block5.down, unet_block5.up

    def forward(self, x):
        """Standard forward"""
        # downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        under_activate_features = [d1, d2, d3, d4, d5]

        # upsample
        u5 = self.up5(d5)
        u4 = self.up4(torch.cat([u5, d4], 1))
        u3 = self.up3(torch.cat([u4, d3], 1))
        u2 = self.up2(torch.cat([u3, d2], 1))
        u1 = self.up1(torch.cat([u2, d1], 1))
        return u1
#         return u1, under_activate_features

class UnetSkipConnectionBlock(nn.Module):
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
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]
        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)
