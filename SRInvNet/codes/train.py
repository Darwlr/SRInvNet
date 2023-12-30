import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import codes.options.options as option
from codes.utils import util
from codes.models import create_model
from codes.data import srdata
from torch.utils.data import DataLoader
import os
import numpy as np


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


root_dir = './results/'
def get_path(*subdir):
    return os.path.join(root_dir, *subdir)

def save_results(save_dir_suffix, filename, save_list, params=None):
    save_name = filename

    filename = get_path(
        f'results-{save_dir_suffix}',
        f'{save_name}.dat')
    os.makedirs(get_path('results-{}'.format(save_dir_suffix)), exist_ok=True)

    for v in save_list:
        sa = v[0].cpu().numpy()
        [ma, mi] = [params[0].cpu().numpy(), params[1].cpu().numpy()]

        sa = sa * (ma - mi) + mi
        sa = np.array(sa, dtype=np.float32)

        sa = np.rot90(sa, -3)
        sa.tofile(filename)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='./options/train/train_InvNet.yml', help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    #### create train and val dataloader
    train_set = srdata.SRData(opt, train=True)
    train_loader = DataLoader(
        train_set, batch_size=opt['datasets']['batch_size'],
        shuffle=True, num_workers=opt['datasets']['n_workers']
    )
    train_size = int(3200 / opt['datasets']['batch_size'])  # 800
    total_iters = int(opt['train']['niter'])  # 160000
    total_epochs = int(math.ceil(total_iters / train_size))  # 200

    val_test = srdata.SRData(opt, train=False)
    val_loader = DataLoader(
        val_test, batch_size=1, shuffle=True)

    # #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, (train_data, filename, params) in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)

                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
            # if rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for (val_data, filename, params) in val_loader:
                    idx += 1

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    denoised_img = visuals['Denoised']
                    gt_img = visuals['GT']
                    noisy_img = visuals['Noisy']

                    lr_img = visuals['LR']
                    gtl_img = visuals['LR_ref']

                    # Save Denoised images
                    save_list = [denoised_img]
                    save_results("denoised", filename[0], save_list, params)

                    save_list = [gt_img]
                    save_results("cleanHR", filename[0], save_list, params)

                    save_list = [noisy_img]
                    save_results("noisy", filename[0], save_list, params)

                    save_list = [lr_img]
                    save_results("LR", filename[0], save_list, params)

                    save_list = [gtl_img]
                    save_results("LR_ref", filename[0], save_list, params)

                    # calculate PSNR
                    crop_size = opt['scale']
                    cropped_denoised_img = denoised_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_denoised_img, cropped_gt_img)

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}.'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}.'.format(
                    epoch, current_step, avg_psnr))

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
