# -*- coding: UTF-8 -*-
"""
@Function:由于不能使用与文件夹名称同名的py文件名，因此使用这个py文件来运行
@File: evaluation.py
@Date: 2021/4/12 14:41 
@Author: Hever
"""


import os
import cv2
import numpy as np
import re
import torch
from options.test_options import TestOptions
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def evaluation(opt, test_output_dir='test_latest'):
    # 初始化
    image_size = (opt.crop_size, opt.crop_size)

    if opt.load_iter != 0:
        result_image_dir = os.path.join(opt.results_dir, opt.name, 'test_latest_iter' + str(opt.load_iter) + '/images')
    else:
        # in是为了训练时测试
        if 'result' in test_output_dir:
            result_image_dir = os.path.join(test_output_dir, 'images')
        else:
            result_image_dir = os.path.join(opt.results_dir, opt.name, test_output_dir, 'images')
    if opt.target_gt_dir is not None:
        gt_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir)
        gt_mask_image_dir = os.path.join(opt.dataroot, opt.target_gt_dir + '_mask')
    else:
        gt_image_dir = os.path.join(opt.dataroot, 'target_gt')
        gt_mask_image_dir = os.path.join(opt.dataroot, 'target_gt_mask')

    # 可视化
    post_output_dir = os.path.join(opt.results_dir, opt.name, 'post_image')
    if not os.path.isdir(post_output_dir):
        os.mkdir(post_output_dir)
    image_name_list = os.listdir(result_image_dir)
    # 前期准备
    sum_psnr = sum_ssim = count = 0
    dict_sum_ssim = {}
    dict_sum_max = {}
    end_word = 'fake_B.png'
    # if 'pix2pix' in opt.model or 'cycle' in opt.model:
    #     end_word = 'fake_B.png'
    # else:
    #     end_word = 'fake_TB.png'
    for image_name in image_name_list:
        # TODO:对于cycle应该是fake_B.png
        if not image_name.endswith(end_word):
            continue
        # 初始化操作
        count += 1
        image_num = re.findall(r'[0-9]+', image_name)[0]
        gt_image_name = image_num + '.png'
        image_path = os.path.join(result_image_dir, image_name)
        gt_image_path = os.path.join(gt_image_dir, gt_image_name)
        mask_path = os.path.join(gt_mask_image_dir, gt_image_name)
        # 读取图像
        gt_image = cv2.imread(gt_image_path)
        gt_image = cv2.resize(gt_image, image_size)
        image = cv2.imread(image_path)

        # 读取mask并进行预处理
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255
        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]

        mask_image = image * mask
        gt_image = gt_image * mask

        cv2.imwrite(os.path.join(post_output_dir, image_name), mask_image)

        if count == 10 or count % 100 == 0:
            print(count)

        # 评价
        ssim = structural_similarity(mask_image, gt_image, data_range=255, multichannel=True)
        psnr = peak_signal_noise_ratio(gt_image, mask_image, data_range=255)

        sum_ssim += ssim
        sum_psnr += psnr

        # if not dict_sum_ssim.get(temp_code):
        #     dict_sum_ssim[temp_code] = 0
        # if not dict_sum_max.get(temp_code):
        #     dict_sum_max[temp_code] = (0, 0)
        # if dict_sum_max[temp_code][1] < ssim:
        #     dict_sum_max[temp_code] = (image_name, ssim)
        # # dict_sum_ssim[temp_code] += ssim
        # dict_sum_ssim[temp_code] += ssim

    # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #     f.write('%f,%f\n' % (sum_ssim / count, sum_psnr / count))
    print('ssim', sum_ssim / count)
    print('psnr', sum_psnr / count)
    return sum_ssim / count

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    evaluation(opt)