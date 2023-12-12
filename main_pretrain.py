# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
from random import shuffle
import numpy as np
import os
import time
from pathlib import Path
from torch.autograd import Variable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import MyDataset

from engine_pretrain import train_one_epoch
import utils
import torch.nn as nn


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.80, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--decoder_embed_dim', default=128, type=int)
    parser.add_argument('--decoder_depth', default=8, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='/scratch/project_2006419/rPPG-MAE/pretrain_VIPLST',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/scratch/project_2006419/rPPG-MAE/pretrain_VIPLST',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')#共享内存，将数据直接映射到GPU
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')#加--dist_on_itp则为true
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')
     #数据集
    parser.add_argument('--dataname', type=str, default="VIPL", help='log and save model name')
    parser.add_argument('--STMap_name1', type=str, default='STMap_RGB_Align_CSI.png')
    parser.add_argument('--STMap_name2', type=str, default='STMap_YUV_Align_CSI_CHROM.png')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=224,
                        help='the num of frames')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('-rT', '--reTrain', dest='reTrain', type=int, default=0,
                        help='Load model')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=1,
                        help='re Data')
    parser.add_argument('--log', type=str, default="pretrain_VIPLST", help='log and save model name')
    parser.add_argument('--checkpoint', type=str, default='/scratch/project_2006419/rPPG-MAE/pretrain_VIPLST/checkpoint-150.pth', help='checkpoint path')
    parser.add_argument('--loss_type', type=str, default='CEP', help='loss type')
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--semi', type=str, default='', help='if semi-supervised')

    return parser


def main(args):
    if args.dataname=='VIPL':
        fileRoot = r'/scratch/project_2006419/data/VIPL_processed'
        saveRoot = r'/scratch/project_2006419/data/VIPL_Index/VIPL_STMap_test' + str(args.fold_num) + str(args.fold_index)
        # saveRoot = r'/scratch/project_2006419/data/VIPL_Index/VIPL_STMap50'
    if args.dataname=='PURE':
        fileRoot = r'/scratch/project_2006419/data/PURE_ST/PUREa'
        saveRoot = r'/scratch/project_2006419/data/PURE_Index/PURE_STMap50'
    if args.dataname=='UBFC':
        fileRoot = r'/scratch/project_2006419/data/UBFC_STMap/UBFC_ST'
        saveRoot = r'/scratch/project_2006419/data/UBFC_Index/UBFC_STMap50'
    
     # 图片参数
    frames_num = args.frames_num
    dataname=args.dataname
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(frames_num, frames_num))
    
    # misc.init_distributed_mode(args)
    print('Not using distributed mode')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 数据集
    # print(os.environ['CUDA_VISIBLE_DEVICES'])
    if args.reData == 1:
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=5, fold_index=0)
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'STMap.png', 5, frames_num)
        Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'STMap.png', 5, frames_num)
   
    dataset_train = MyDataset.Data_DG(root_dir=(saveRoot + '_Train'),dataName=dataname,STMap1 =args.STMap_name1, STMap2 =args.STMap_name2, \
        in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    

    print('trainLen:', len(dataset_train))

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](decoder_embed_dim=args.decoder_embed_dim,
                                                    decoder_depth=args.decoder_depth,
                                                    in_chans=args.in_chans,
                                                    norm_pix_loss=args.norm_pix_loss)
  
    if args.reTrain == 1:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        # for k,v in checkpoint_model.items():
        #     print(k)
        print('load model ...' )
        

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    if args.reTrain:
      optimizer.load_state_dict(checkpoint['optimizer'])
      loss_scaler.load_state_dict(checkpoint['scaler'])
    # print(optimizer)
    

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.reTrain:
        checkpoint_path = args.checkpoint
        output_dir_path = args.output_dir
        log_dir_path = args.log_dir
        log_path = args.log
        decoder_embed_dim = args.decoder_embed_dim
        decoder_depth = args.decoder_depth
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        args = checkpoint['args']
        args.start_epoch=checkpoint['epoch']+1
        args.reTrain = 1
        args.reData = 0
        args.checkpoint= checkpoint_path
        args.output_dir = output_dir_path
        args.log_dir = log_dir_path
        args.log = log_path
        args.decoder_embed_dim = decoder_embed_dim
        args.decoder_depth = decoder_depth
    else:
      if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
