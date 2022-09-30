# -*- coding: UTF-8 -*-
"""
@Function:
@File: HFC_filter.py
@Date: 2021/7/26 15:02 
@Author: Hever
"""
from torch import nn
from torch.nn import functional as F
import torch
import cv2


class LPLSPyramid(nn.Module):
    def __init__(self, filter_width=13, nsig=10, sub_mask=False, ratio=4, is_clamp=True, insert_blur=True, insert_level=False):
        """
        ratio：放大特征图，结合is_clamp使用，去除部分值太小和太大的值，提升对比度
        insert_blur：bool，是否对下采样又上采样的特征做模糊，可以保留更多高频成分
        insert_level：是否根据特征图大小改变insert_blur的高斯模糊核大小，需要设置合理的filter_width和nsig，否则使用默认的5 1，
        filter width越大，保留的高频越多；nsig越大，保留的高频越多
        """
        super(LPLSPyramid, self).__init__()
        self.gaussian_filter1 = Gaussian_kernel(
            filter_width, nsig=nsig)
        self.gaussian_filter2 = Gaussian_kernel(
            int(filter_width / 2) + int(filter_width / 2) % 2 + 1, nsig=nsig/2)
        self.gaussian_filter3 = Gaussian_kernel(
            int(filter_width / 4) + int(filter_width / 4) % 2 + 1, nsig=nsig/4)
        # self.blur = Gaussian_blur_kernel()
        self.blur = Gaussian_kernel(5, nsig=1)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp
        self.insert_blur = insert_blur
        self.insert_level = insert_level

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)

        # down
        down1 = self.downsample(self.blur(x))
        mask1 = torch.nn.functional.interpolate(mask, scale_factor=1/2, mode='bilinear')
        down2 = self.downsample(self.blur(down1))
        mask2 = torch.nn.functional.interpolate(mask1, scale_factor=1/2, mode='bilinear')
        down3 = self.downsample(self.blur(down2))
        mask3 = torch.nn.functional.interpolate(mask2, scale_factor=1/2, mode='bilinear')

        # up
        up1 = torch.nn.functional.interpolate(down1, scale_factor=2, mode='bilinear')
        up2 = torch.nn.functional.interpolate(down2, scale_factor=2, mode='bilinear')
        up3 = torch.nn.functional.interpolate(down3, scale_factor=2, mode='bilinear')

        # low
        if self.insert_blur:
            if self.insert_level:
                x_low = self.gaussian_filter1(up1)
                down1_low = self.gaussian_filter2(up2)
                down2_low = self.gaussian_filter3(up3)
            else:
                x_low = self.blur(up1)
                down1_low = self.blur(up2)
                down2_low = self.blur(up3)
        else:
            x_low = up1
            down1_low = up2
            down2_low = up3

        # high
        h1 = self.sub_low_freq(x, x_low, mask)
        h2 = self.sub_low_freq(down1, down1_low, mask1)
        h3 = self.sub_low_freq(down2, down2_low, mask2)

        if self.sub_mask:
            down3 = (down3 + 1) * mask3 - 1
        return [h1, h2, h3, down3]

    def sub_low_freq(self, x, low, mask):
        res = self.ratio * (x - low)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1
        return res


class LP5Layer(nn.Module):
    def __init__(self, filter_width=13, nsig=10, sub_mask=False, ratio=4, is_clamp=True, insert_blur=True, insert_level=False):
        """
        ratio：放大特征图，结合is_clamp使用，去除部分值太小和太大的值，提升对比度
        insert_blur：bool，是否对下采样又上采样的特征做模糊，可以保留更多高频成分
        insert_level：是否根据特征图大小改变insert_blur的高斯模糊核大小，需要设置合理的filter_width和nsig，否则使用默认的5 1，
        filter width越大，保留的高频越多；nsig越大，保留的高频越多
        """
        super(LP5Layer, self).__init__()
        self.gaussian_filter1 = Gaussian_kernel(
            filter_width, nsig=nsig)
        self.gaussian_filter2 = Gaussian_kernel(
            int(filter_width / 2) + int(filter_width / 2) % 2 + 1, nsig=nsig/2)
        self.gaussian_filter3 = Gaussian_kernel(
            int(filter_width / 4) + int(filter_width / 4) % 2 + 1, nsig=nsig/4)
        self.gaussian_filter4 = Gaussian_kernel(
            int(filter_width / 6) + int(filter_width / 6) % 2 + 1, nsig=nsig / 6)
        # self.blur = Gaussian_blur_kernel()
        self.blur = Gaussian_kernel(5, nsig=1)
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.max = 1.0
        self.min = -1.0
        self.ratio = ratio
        self.sub_mask = sub_mask
        self.is_clamp = is_clamp
        self.insert_blur = insert_blur
        self.insert_level = insert_level

    def median_padding(self, x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].view([batch_size, -1]).median(dim=1).values.view(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1)
        median_tensor = median_tensor.unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def forward(self, x, mask):
        assert mask is not None
        x = self.median_padding(x, mask)

        # down
        down1 = self.downsample(self.blur(x))
        mask1 = torch.nn.functional.interpolate(mask, scale_factor=1/2, mode='bilinear')
        down2 = self.downsample(self.blur(down1))
        mask2 = torch.nn.functional.interpolate(mask1, scale_factor=1/2, mode='bilinear')
        down3 = self.downsample(self.blur(down2))
        mask3 = torch.nn.functional.interpolate(mask2, scale_factor=1/2, mode='bilinear')
        down4 = self.downsample(self.blur(down3))
        mask4 = torch.nn.functional.interpolate(mask3, scale_factor=1 / 2, mode='bilinear')

        # up
        up1 = torch.nn.functional.interpolate(down1, scale_factor=2, mode='bilinear')
        up2 = torch.nn.functional.interpolate(down2, scale_factor=2, mode='bilinear')
        up3 = torch.nn.functional.interpolate(down3, scale_factor=2, mode='bilinear')
        up4 = torch.nn.functional.interpolate(down4, scale_factor=2, mode='bilinear')

        # low
        if self.insert_blur:
            if self.insert_level:
                x_low = self.gaussian_filter1(up1)
                down1_low = self.gaussian_filter2(up2)
                down2_low = self.gaussian_filter3(up3)
                down3_low = self.gaussian_filter4(up4)
            else:
                x_low = self.blur(up1)
                down1_low = self.blur(up2)
                down2_low = self.blur(up3)
                down3_low = self.blur(up4)
        else:
            x_low = up1
            down1_low = up2
            down2_low = up3
            down3_low = up4

        # high
        h1 = self.sub_low_freq(x, x_low, mask)
        h2 = self.sub_low_freq(down1, down1_low, mask1)
        h3 = self.sub_low_freq(down2, down2_low, mask2)
        h4 = self.sub_low_freq(down3, down3_low, mask3)

        if self.sub_mask:
            down4 = (down4 + 1) * mask4 - 1
        return [h1, h2, h3, h4, down4]

    def sub_low_freq(self, x, low, mask):
        res = self.ratio * (x - low)
        if self.is_clamp:
            res = torch.clamp(res, self.min, self.max)
        if self.sub_mask:
            res = (res + 1) * mask - 1
        return res

def get_kernel(kernel_len=16, nsig=10.0):  # nsig 标准差 ，kernlen=16核尺寸
    GaussianKernel = cv2.getGaussianKernel(kernel_len, nsig) \
                     * cv2.getGaussianKernel(kernel_len, nsig).T
    return GaussianKernel


class Gaussian_blur_kernel(nn.Module):
    def __init__(self):
        super(Gaussian_blur_kernel, self).__init__()
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(3, 1, 1, 1)
        # kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        x_output = F.conv2d(x, self.weight, groups=x.shape[1])
        return x_output


class Gaussian_kernel(nn.Module):
    def __init__(self,
                 # device,
                 kernel_len, nsig=20.0):
        super(Gaussian_kernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = get_kernel(kernel_len=kernel_len, nsig=nsig)  # 获得高斯卷积核
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展两个维度
        kernel = kernel.repeat(3, 1, 1, 1)
        # self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        # self.w = weight.repeat(3, 1, 1, 1)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len/2))

    def forward(self, x):  # x1是用来计算attention的，x2是用来计算的Cs
        x = self.padding(x)
        # 对三个channel分别做卷积
        # res = []
        # for i in range(x.shape[1]):
        #     res.append(F.conv2d(x[:, i:i+1], self.weight))
        # x_output = torch.cat(res, dim=1)
        x_output = F.conv2d(x, self.weight, groups=x.shape[1])
        return x_output

