# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import utils


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    loss_func_rPPG = utils.P_loss3().to(device)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples1, bvp_rel, HR_rel) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        if args.loss_type =='CEP':
            samples1 = samples1.to(device, non_blocking=True)
            # samples2 = samples2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, Wave_pr, _ = model(samples1, mask_ratio=args.mask_ratio)

        if args.loss_type == 'rppg':
            from torch.autograd import Variable
            loss_func_rPPG = utils.P_loss3().to(device)
            samples = Variable(samples).float().to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                _ , Wave_pr, _ = model(samples, mask_ratio=args.mask_ratio)
            loss = torch.zeros(1).to(device)
            for chan in range(args.in_chans):
                for width in range(224):
                    loss = loss + loss_func_rPPG(Wave_pr[:, chan, width, :].unsqueeze(dim=1), samples[:, chan, width, :].unsqueeze(dim=1))
            loss = loss/(224*args.in_chans) 
        if args.loss_type == 'SP':
            from torch.autograd import Variable
            # loss_func_rPPG = utils.P_loss3().to(device)
            loss_func_SP = utils.SP_loss_pretrain(device, low_bound=36,high_bound=240, clip_length=args.frames_num).to(device)
            samples = Variable(samples).float().to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                _ , Wave_pr, _ = model(samples, mask_ratio=args.mask_ratio)
            loss1 = torch.zeros(1).to(device)
            loss2 = torch.zeros(1).to(device)
            for chan in range(args.in_chans):
                for width in range(224):
                    # loss1 = loss1 + loss_func_rPPG(Wave_pr[:, chan, width, :].unsqueeze(dim=1), samples[:, chan, width, :].unsqueeze(dim=1))
                    loss2 = loss2 + loss_func_SP(Wave_pr[:, chan, width, :].unsqueeze(dim=1), samples[:, chan, width, :].unsqueeze(dim=1))
            loss = (loss2)/(224*args.in_chans) 
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}