# -*- coding: UTF-8 -*-
import torch
import itertools
from .base_model import BaseModel
from . import networks
from util import util
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from models.layer.HFC_filter import HFCFilter
from models.layer.lpls_pyramid import LP5Layer
from models.layer.spp_layer import SpatialPyramidPooling


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    return (hfc + 1) * mask - 1

def compute_cos_sim(input, target):
    loss = (1 - torch.cosine_similarity(x1=input, x2=target, dim=1))
    loss = torch.sum(loss) / input.shape[0]
    return loss

class PCENetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='pce_backbone', dataset_mode='fiq_basic', no_dropout=True)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0)
            parser.add_argument('--feature_loss', type=str, default='cos')
            parser.add_argument('--lambda_feature', type=float, default=10)
            parser.add_argument('--lambda_layer', nargs='+', type=float, default=[1, 1, 1, 1, 1])

            # parser.add_argument('--lambda_pyramid', type=float, default=1.0)
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G', 'G_feature', 'G_feature_vis']

        self.visual_names_train = ['real_A', 'real_A_input', 'down_h1', 'down_h2', 'down_h3', 'down4', 'fake_B', 'real_B']
        self.visual_names_test = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.lpls_pyramid = LP5Layer(5, 1, sub_mask=True, insert_level=False).to(self.device)
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            if self.opt.feature_loss == 'l2':
                self.criterionFeature = torch.nn.MSELoss()
            elif self.opt.feature_loss == 'cos':
                self.criterionFeature = compute_cos_sim
            else:
                self.criterionFeature = torch.nn.L1Loss()
            self.mean = torch.mean
            self.DR_batch_size = opt.DR_batch_size
            self.batch_size = opt.batch_size
            self.visual_names = self.visual_names_train
            self.relu = torch.nn.ReLU()
            self.visual_names = self.visual_names_train
            self.spp = SpatialPyramidPooling(levels=[2, 4, 8])
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        else:
            self.visual_names = self.visual_names_test


    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.mask = input['A_mask'].to(self.device)
        # # self.real_A_lpls1 = hfc_mul_mask(self.hfc_filter, self.real_A, self.A_mask)
        # self.image_paths = input['A_path']
        # self.real_A_pyramid = self.lpls_pyramid(self.real_A, self.mask)
        # pass
        AtoB = self.opt.direction == 'AtoB'
        if isTrain:
            self.real_A_list = input['A_list' if AtoB else 'B_list'].to(self.device)
            self.real_B_list = input['B_list' if AtoB else 'A_list'].to(self.device)
            self.mask_list = input['A_mask_list'].to(self.device)
            self.real_A = torch.cat([t for t in self.real_A_list])
            self.real_B = torch.cat([t for t in self.real_B_list])
            self.mask = torch.cat([t for t in self.mask_list])
            # 对batch list进行处理
            self.image_paths = input['image_path_list']
            self.paths = []
            for i in range(len(self.image_paths[0])):
                for j in range(len(self.image_paths)):
                    self.paths.append(self.image_paths[j][i])
            self.real_A_pyramid = self.lpls_pyramid(self.real_A, self.mask)
            self.real_B_pyramid = self.lpls_pyramid(self.real_B, self.mask)
        else:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.mask = input['A_mask'].to(self.device)
            self.image_paths = input['A_path']
            self.real_A_pyramid = self.lpls_pyramid(self.real_A, self.mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B, self.real_A_feature = self.netG(self.real_A_pyramid, need_feature=True)
        self.fake_B = (self.fake_B + 1) * self.mask - 1
        # self.fake_B = self.netG(self.real_A_pyramid)  # G(A)
        # self.fake_B = (self.fake_B + 1) * self.A_mask - 1

        self.this_batch_size = len(self.fake_B) // self.DR_batch_size
        self.real_A_relu_spp_features = [self.spp(self.relu(f)) for f in self.real_A_feature]
        # 计算每个batch的均值，将batch进行分装
        self.real_A_relu_spp_features_mean = []
        for l in range(len(self.real_A_relu_spp_features)):
            t = []
            for i in range(self.this_batch_size):
                t += [self.mean(self.real_A_relu_spp_features[l][i*self.DR_batch_size:(i+1)*self.DR_batch_size],
                             dim=0, keepdim=True)] * self.DR_batch_size
            self.real_A_relu_spp_features_mean.append(torch.cat(t, dim=0))

    def compute_visuals(self):
        x_high, down_h1, down_h2, down_h3, down4 = self.real_A_pyramid
        self.real_A_input = torch.nn.functional.interpolate(x_high[:, :3], size=(256, 256), mode='nearest')
        self.down_h1 = torch.nn.functional.interpolate(down_h1[:, :3], size=(256, 256), mode='nearest')
        self.down_h2 = torch.nn.functional.interpolate(down_h2[:, :3], size=(256, 256), mode='nearest')
        self.down_h3 = torch.nn.functional.interpolate(down_h3[:, :3], size=(256, 256), mode='nearest')
        self.down4 = torch.nn.functional.interpolate(down4[:, :3], size=(256, 256), mode='nearest')

    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # for visualization
            self.fake_B = self.netG(self.real_A_pyramid)  # G(A)
            self.fake_B = (self.fake_B + 1) * self.mask - 1
            if self.opt.eval_when_train:
                self.eval_when_train()

    def eval_when_train(self):
        self.ssim = self.psnr = 0
        for i in range(len(self.fake_B)):
            self.fake_B_im = util.tensor2im(self.fake_B[i:i+1])
            self.real_B_im = util.tensor2im(self.real_B[i:i+1])
            self.ssim += structural_similarity(self.real_B_im, self.fake_B_im, data_range=255, multichannel=True)
            self.psnr += peak_signal_noise_ratio(self.real_B_im, self.fake_B_im, data_range=255)

    def train(self):
        """Make models eval mode during test time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def backward_G(self):
        self.loss_G_feature = self.loss_G_idt = 0
        self.loss_G_feature_vis = 0

        # 计算每层的损失
        for l_mean, l, lambda_l in zip(self.real_A_relu_spp_features_mean, self.real_A_relu_spp_features, self.opt.lambda_layer):
            loss_temp = self.criterionFeature(l_mean, l)
            self.loss_G_feature_vis += loss_temp
            self.loss_G_feature += loss_temp * lambda_l * self.opt.lambda_feature

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_L1 + self.loss_G_feature + self.loss_G_idt
        self.loss_G.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        self.compute_visuals()

