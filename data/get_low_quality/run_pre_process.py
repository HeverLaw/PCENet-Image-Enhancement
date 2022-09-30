from utils_de import *
import glob
import os
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dsize = (512,512)
def process(save_path, image_list):
    if not os.path.isdir(os.path.join(save_path, 'image')):
        os.mkdir(os.path.join(save_path, 'image'))
    if not os.path.isdir(os.path.join(save_path, 'mask')):
        os.mkdir(os.path.join(save_path, 'mask'))
    for image_path in image_list:
        dst_image = image_path.split('\\')[-1]
        dst_image_path = os.path.join(save_path, 'image', dst_image.replace('.jpeg', '.png'))
        dst_mask_path = os.path.join(save_path, 'mask', dst_image.replace('.jpeg', '.png'))
        if os.path.exists(dst_image_path):
            print('continue...')
            continue
        try:
            img = imread(image_path)
            img, mask = preprocess(img)
            img = cv2.resize(img, dsize)
            mask = cv2.resize(mask, dsize)
            imwrite(dst_image_path, img)
            imwrite(dst_mask_path, mask)
        except:
            print(image_path)
            continue


def mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    image_root = r'./image/high_quality_image_original'
    save_root = r'./image/high_quality_image_pre_process'
    mkdir(save_root)
    image_list = glob.glob(os.path.join(image_root, '*.png'))
    image_list += glob.glob(os.path.join(image_root, '*.jpeg'))
    process(save_root, image_list)


        





