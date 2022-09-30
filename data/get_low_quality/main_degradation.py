import os
import glob
from multiprocessing.pool import Pool

import numpy as np
from utils_de import imread, imwrite
from PIL import Image
from degrad_de import *
import json
import random
import shutil

random.seed(2021)
np.random.seed(2021)
sizeX = 512
sizeY = 512


type_map = ['001', '010', '100', '011', '101', '110', '111']
num_type = 16
clear_image_dir = './image/high_quality_image_pre_process/image'
clear_image_mask_dir = './image/high_quality_image_pre_process/mask'

output_dir = './image/low_quality_image'
output_mask_dir = './image/low_quality_mask'
output_param_dir = './image/low_quality_param'

# '111' means: DE_BLUR, DE_SPOT, DE_ILLUMINATION

def mkdirs(*dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)


def generate_type_list(num_type):
    type_list = []
    if num_type >= len(type_map):
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    else:
        for i in range(num_type):
            t = random.randint(0, len(type_map) - 1)
            type_list.append(type_map[t])
    return type_list


def degradation(image_dir, image_mask_dir, output_dir, output_param_dir, num_type=16):
    image_list = os.listdir(image_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(image_mask_dir, image_name)

        img = Image.open(image_path).convert('RGB')
        shutil.copy(mask_path, os.path.join(output_mask_dir, image_name))
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((sizeX, sizeY), Image.BICUBIC)
        mask = np.expand_dims(mask, axis=2)
        mask = np.array(mask, np.float32).transpose(2, 0, 1)/255.0
        type_list = generate_type_list(num_type)
        for i, t in enumerate(type_list):
            r_img, r_params = DE_process(img, mask, sizeX, sizeY, t)
            dst_img = os.path.join(output_dir, '{}-{}.png'.format(image_name.split('.')[0], i))
            imwrite(dst_img, r_img)
            param_dict = {
                'name': image_name,
                'type': t,
                'params': r_params
            }
            with open(os.path.join(output_param_dir, '{}-{}.json'.format(image_name.split('.')[0], i)), 'w') as json_file:
                json.dump(param_dict, json_file)

        
if __name__=="__main__":
    mkdirs(output_dir, output_param_dir, output_mask_dir)
    degradation(clear_image_dir, clear_image_mask_dir, output_dir, output_param_dir, num_type=num_type)

