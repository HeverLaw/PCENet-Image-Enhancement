import time, os
import copy
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset, create_target_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
from util.metric_logger import MetricLogger
from model_eval.evaluation import evaluation

def model_test(testOpt, testDataset, model, web_dir, eval_test=False):
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (testOpt.name, testOpt.phase, testOpt.epoch))
    if eval_test:
        model.eval()
    for i, data in enumerate(testDataset):
        if i >= testOpt.num_test:  # only apply our model to opt.num_test images.
            print('process finish:', i)
            break
        model.set_input(data, isTrain=False)  # unpack data from data loader

        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    model.train()
    webpage.save()  # save the HTML


def model_eval(testOpt, testDataset, model, eval_test=False, save_result=False, web_dir='./result'):
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (testOpt.name, testOpt.phase, testOpt.epoch))
    sum_ssim = sum_psnr = 0
    if eval_test:
        model.eval()
    for i, data in enumerate(testDataset):
        if i >= testOpt.num_test:  # only apply our model to opt.num_test images.
            print('process finish:', i)
            break
        model.set_input(data, isTrain=False)  # unpack data from data loader
        model.test()  # run inference
        sum_ssim += model.ssim
        sum_psnr += model.psnr
        if save_result:
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    if save_result:
        webpage.save()  # save the HTML
    # with open(os.path.join(opt.results_dir, 'log', opt.name + '.csv'), 'a') as f:
    #     f.write('%f,%f' % (sum_ssim / len(testDataset.dataset), sum_psnr / len(testDataset.dataset)))
    print('Evaluation result: ssim: {:.3f}, psnr: {:.2f}'.format(
        sum_ssim / len(testDataset.dataset), sum_psnr / len(testDataset.dataset)
    ))
    model.train()
    return sum_ssim / len(testDataset.dataset), sum_psnr / len(testDataset.dataset)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    meters = MetricLogger(delimiter="  ")

    total_iters = 0                # the total number of training iterations
    max_ssim = max_ssim_iter = 0
    if opt.pretrain_model != '':
        print('loading model from {}'.format(opt.pretrain_model))
        model.load_pretrain_model(opt)
    # ----------为了测试的初始化-------------
    testOpt = copy.deepcopy(opt)
    # testOpt = TestOptions().parse()  # get test options
    testOpt.phase = 'test'
    testOpt.isTrain = False
    testOpt.num_threads = 0  # test code only supports num_threads = 1
    testOpt.batch_size = 1  # test code only supports batch_size = 1
    testOpt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    testOpt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    testOpt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    testOpt.load_size = testOpt.crop_size = testOpt.test_crop_size
    testOpt.dataset_mode = testOpt.test_dataset_when_train
    testDataset = create_dataset(testOpt)  # create a dataset given opt.dataset_mode and other options
    if opt.eval_when_train:
        evalOpt = copy.deepcopy(opt)
        evalOpt.phase = 'eval'
        evalOpt.isTrain = False
        evalOpt.num_threads = 0  # test code only supports num_threads = 1
        evalOpt.batch_size = 8  # test code only supports batch_size = 1
        evalOpt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        evalOpt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
        evalOpt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
        evalOpt.load_size = evalOpt.crop_size = evalOpt.test_crop_size
        evalOpt.dataset_mode = evalOpt.test_dataset_when_train
        evalDataset = create_dataset(evalOpt)  # create a dataset given opt.dataset_mode and other options
    # # ------------测试模型-----------------
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.在每个epoch开始前更新

        for i, data_source in enumerate(dataset):
            data = data_source
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data, isTrain=True)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # 直接将网络初始化时希望可视化的参数送到visualizer
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印loss和可视化loss
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                # 第二个epoch重新设置loss
                if epoch == 2:
                    meters = MetricLogger(delimiter="  ")
                meters.update(**losses)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_meters_losses(epoch, epoch_iter, losses, t_comp, t_data, meters)
                if opt.display_id > 0:
                    # TODO:可视化时适当修改dataset_size
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # 保存网络，根据图像的的iter来，而不是epoch，基本不用，因为iter不是累加的
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()
        # ------------测试模型-----------------
        if opt.eval_when_train and (epoch % opt.test_freq == 0 or epoch == 2):
            print('Evaluating...')
            eval_web_dir = os.path.join(evalOpt.results_dir, evalOpt.name,
                                        '{}_{}'.format(evalOpt.phase, 'latest'))
            eval_web_dir = '{:s}_iter{:d}_eval'.format(eval_web_dir, epoch)
            # if you want to observe the evaluation result, please set save_result as True
            eval_ssim, eval_psnr = model_eval(evalOpt, evalDataset, model, evalOpt.eval_test, save_result=False,
                                              web_dir=eval_web_dir)
        if (opt.test_when_train and epoch % opt.test_freq == 0) or (opt.test_when_train and epoch == 2):
            test_web_dir = os.path.join(testOpt.results_dir, testOpt.name,
                                        '{}_{}'.format(testOpt.phase, 'latest'))
            test_web_dir = '{:s}_iter{:d}'.format(test_web_dir, epoch)
            print('creating web directory', test_web_dir)
            model_test(testOpt, testDataset, model, test_web_dir, testOpt.eval_test)
            if opt.test_ssim_psnr:
                ssim = evaluation(testOpt, test_web_dir)

        # 保存网络
        if epoch % opt.save_epoch_freq == 0 or epoch == 100:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    model.save_networks('latest')

