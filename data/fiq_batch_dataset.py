import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class FiqBatchDataset(BaseDataset):
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
        #
        self.isTrain = opt.isTrain
        self.source_clear_num_images = opt.source_clear_num_images
        # self.need_mask = opt.need_mask
        self.DR_batch_size = opt.DR_batch_size
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        image_dir_name = 'source_image' if self.isTrain else 'target_image'
        gt_dir_name = 'source_gt' if self.isTrain else 'target_gt'
        image_mask_dir_name = 'source_mask' if self.isTrain else 'target_mask'
        # directory
        self.image_dir = os.path.join(opt.dataroot, image_dir_name)
        self.gt_dir = os.path.join(opt.dataroot, gt_dir_name)
        self.image_mask_dir = os.path.join(opt.dataroot, image_mask_dir_name)

        # 设置clear图像数，设置DR的图像数
        self.image_indexes = [str(i) for i in range(1, self.source_clear_num_images)]
        self.image_names = os.listdir(self.image_dir)
        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))  # get image paths
        self.DR_num = int(len(self.image_paths) / self.source_clear_num_images)
        # self.gt_paths = sorted(make_dataset(self.gt_dir, opt.max_dataset_size))  # get image paths
        # self.mask_paths = sorted(make_dataset(self.mask_dir, opt.max_dataset_size))  # get image paths


    def __getitem__(self, index):
        A_list = []
        B_list = []
        mask_list = []
        image_path_list = []

        batch_index_list = [i for i in range(self.DR_num)]
        random.shuffle(batch_index_list)
        image_name_prefix = os.path.split(self.image_names[index % self.source_clear_num_images])[-1].split('-')[0].replace('.png', '')
            # os.path.split(image_path)[-1].split('-')[0].replace('.png', '') + '.png'

        gt_name = '{}.png'.format(image_name_prefix)
        gt_path = os.path.join(self.gt_dir, gt_name)
        B = Image.open(gt_path).convert('RGB')
        transform_params = get_params(self.opt, B.size)
        image_transform, mask_transform = get_transform_six_channel(self.opt, transform_params,
                                                                    grayscale=(self.input_nc == 1))

        B = image_transform(B)
        mask_path = os.path.join(self.image_mask_dir, image_name_prefix + '.png')
        mask = Image.open(mask_path).convert('L')
        mask = mask_transform(mask)
        for i in range(self.DR_batch_size):
            image_name = '{}-{}.png'.format(image_name_prefix, batch_index_list[i])
            # 为了适配target
            image_path = os.path.join(self.image_dir, image_name)
            A = Image.open(image_path).convert('RGB')
            A = image_transform(A)
            mask_list.append(mask.unsqueeze(dim=0))
            A_list.append(A.unsqueeze(dim=0))
            B_list.append(B.unsqueeze(dim=0))
            image_path_list.append((image_path))
        A_list_tensor = torch.cat(A_list, dim=0)
        B_list_tensor = torch.cat(B_list, dim=0)
        mask_list_tensor = torch.cat(mask_list, dim=0)
        return {'A_list': A_list_tensor, 'B_list': B_list_tensor, 'image_path_list': image_path_list,
                'A_mask_list': mask_list_tensor, 'B_mask_list': mask_list_tensor}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return int(len(self.image_paths) / self.DR_num)
