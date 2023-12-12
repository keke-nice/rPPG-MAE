# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit
import MyDataset
import wandb
from torch.autograd import Variable
import scipy.io as io
import utils

from engine_finetune import train_one_epoch, evaluate
from utils_sig import *

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=220, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--dataname', type=str, default="VIPL", help='log and save model name')
    parser.add_argument('--STMap_name1', type=str, default="STMap_RGB_Align_CSI.png", help='log and save model name')
    parser.add_argument('--STMap_name2', type=str, default="STMap_YUV_Align_CSI_CHROM.png", help='log and save model name')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')


    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=224,
                        help='the num of frames')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('--log', type=str, default="supervise_VIT_VIPL_LossCrossEntropy", help='log and save model name')
    parser.add_argument('--loss_type', type=str, default="SP", help='loss type')
    parser.add_argument('--in_chans', type=int, default=3)
    return parser


def main(args):
    # misc.init_distributed_mode(args)

    fileRoot = r'/scratch/project_2006419/data/VIPL_processed'
    saveRoot = r'/scratch/project_2006419/data/VIPL_Index/VIPL_STMap50'
    
    # fileRoot = r'/media/26d532/keke/data/PURE_ST/PUREa'
    # saveRoot = r'/media/26d532/keke/data/PURE_ST/PURE_Index/PURE_STMap50'
    wandb.init(project="VIPL_linprobenew", entity="rppg" ,name =args.log)
    wandb.config = {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size
                    }
    best_mae =20
    frames_num = args.frames_num
    dataname=args.dataname
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    # resize = transforms.Resize(size=(64, frames_num))
    resize = transforms.Resize(size=(frames_num, frames_num))

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # device = 'cpu'
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 数据集
    # if args.reData == 1:
    #     test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=fold_num,fold_index=fold_index)
    #     Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'STMap.png', 5, frames_num)
    #     Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'STMap.png', 5, frames_num)
    # dataset_train = MyDataset.Data_DG(root_dir=(saveRoot + '_Train'),dataName=args.dataname,STMap =args.STMap_name,frames_num=args.frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    # dataset_val = MyDataset.Data_DG(root_dir=(saveRoot + '_Test'),dataName=args.dataname,STMap =args.STMap_name,frames_num=args.frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    # train_loader = DataLoader(train_db, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    # test_loader =torch.utils.data.DataLoader(dataset_val, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)
    dataset_train = MyDataset.Data_DG(root_dir=(saveRoot + '_Train'),dataName=dataname,STMap1 =args.STMap_name1, STMap2 =args.STMap_name2, \
        in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    dataset_val = MyDataset.Data_DG(root_dir=(saveRoot + '_Test'),dataName=dataname,STMap1 =args.STMap_name1, STMap2 =args.STMap_name2, \
        in_chans = args.in_chans, frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    print('trainLen:', len(dataset_train), 'testLen:', len(dataset_val))
    print('fold_num:', args.fold_num, 'fold_index', args.fold_index)


    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir is not None:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        in_chans=args.in_chans
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
         #test
        model.eval()
        HR_pr_temp = []#测试集所有预测心率
        HR_rel_temp = []
        for step, (data, bvp, HR_rel) in enumerate(data_loader_val):
            data = Variable(data).float().to(device=device)
            bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            bvp = bvp.unsqueeze(dim=1)
            STMap = data[:, :, :, 0:frames_num]
            Wave = bvp[:, :, 0:frames_num]
            b, _, _ = Wave.size()

            outputs = model(data)#[B,220]
            if args.loss_type=='CEP':
                targets = torch.tensor([int(np.round(item.cpu().detach().numpy()))-40 for item in HR_rel]).to(device=device)
                loss = criterion(outputs, targets)
                HR_pr=np.argmax(outputs.data.cpu().numpy(), axis=1)+40#[B]
                HR_pr_temp.extend(HR_pr)
                HR_rel_temp.extend(HR_rel.data.cpu().numpy())
            if args.loss_type=='rppg':
                loss_func_rPPG = utils.P_loss3().to(device)
                rppg = butter_bandpass_batch(outputs.data.cpu().numpy(), lowcut=0.6, highcut=4, fs=30)
                hr_pre, psd_y, psd_x = hr_fft_batch(rppg, fs=30)
                hr_rel, _  , _       = hr_fft_batch(bvp.data.cpu().numpy(), fs=30)
                loss = loss_func_rPPG(outputs.unsqueeze(dim=1), Wave)
                HR_pr_temp.extend(hr_pre)
                HR_rel_temp.extend(hr_rel)
            if args.loss_type=='SP':
                loss_func_SP = utils.SP_loss(device, low_bound=36,high_bound=240, clip_length=args.frames_num).to(device)
                loss, hr_pre= loss_func_SP(outputs.unsqueeze(dim=1), HR_rel)
                HR_pr_temp.extend(hr_pre.data.cpu().numpy())
                HR_rel_temp.extend(HR_rel.data.cpu().numpy())
        print('loss_test: ', loss)
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        wandb.log({"MAE": MAE,'epoch': epoch})
        if best_mae > MAE:
            best_mae = MAE
            io.savemat(args.log+'/' + 'HR_pr.mat', {'HR_pr': HR_pr_temp})#训练结束后保存着所有EPOCHE里效果最好的预测心率
            io.savemat(args.log+'/' + 'HR_rel.mat', {'HR_rel': HR_rel_temp})#保存效果最好的真实心率
            print('save best predict HR')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

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
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
