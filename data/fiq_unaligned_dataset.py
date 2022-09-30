import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class FiqUnalignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.need_mask = opt.need_mask
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        image_dir_name = 'source_image' if self.isTrain else 'target_image'  # A
        gt_dir_name = 'source_gt' if self.isTrain else 'target_gt'  # B
        image_mask_dir_name = 'source_mask' if self.isTrain else 'target_mask'  # A_mask
        gt_mask_dir_name = 'source_mask' if self.isTrain else 'target_gt_mask'  # A_mask
        self.image_dir = os.path.join(opt.dataroot, image_dir_name)  # get the image directory
        self.gt_dir = os.path.join(opt.dataroot, gt_dir_name)  # get the image directory
        self.image_mask_dir = os.path.join(opt.dataroot, image_mask_dir_name)  # get the image directory
        self.gt_mask_dir = os.path.join(opt.dataroot, gt_mask_dir_name)  # get the image directory

        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))  # get image paths
        self.gt_paths = sorted(make_dataset(self.gt_dir, opt.max_dataset_size))  # get image paths
        self.len_gt = len(self.gt_paths)
        # self.mask_paths = sorted(make_dataset(self.mask_dir, opt.max_dataset_size))  # get image paths


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # 为了适配target
        image_name = os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'
        gt_index = random.randint(0, self.len_gt - 1)
        gt_path = self.gt_paths[gt_index]

        A = Image.open(image_path).convert('RGB')
        B = Image.open(gt_path).convert('RGB')

        # w, h = A.size
        # 对输入和输出进行同样的transform（裁剪也继续采用）
        transform_params = get_params(self.opt, A.size)
        image_transform, mask_transform = get_transform_six_channel(self.opt, transform_params, grayscale=(self.input_nc == 1))

        A = image_transform(A)
        B = image_transform(B)
        if self.need_mask:
            image_mask_path = os.path.join(self.image_mask_dir, image_name)
            gt_mask_path = os.path.join(self.gt_mask_dir, image_name)
            image_mask = Image.open(image_mask_path).convert('L')
            gt_mask = Image.open(gt_mask_path).convert('L')
            A_mask = mask_transform(image_mask)
            B_mask = mask_transform(gt_mask)
            return {'A': A, 'B': B, 'A_path': image_path,
                    'A_mask': A_mask, 'B_mask': B_mask}
        return {'A': A, 'B': B, 'A_path': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
