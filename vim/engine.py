# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    epoch_start_time = time.time()
    total_samples = 0
    iter_start_time = time.time()
    iteration = 0
        
    # debug
    # count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count += 1
        # if count > 20:
        #     break

        batch_size = samples.shape[0]
        total_samples += batch_size
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # Calculate throughput (img/sec)
        iter_time = time.time() - iter_start_time
        img_per_sec = 0
        if iter_time > 0:
            img_per_sec = batch_size / iter_time
            metric_logger.update(img_s=img_per_sec)
        iter_start_time = time.time()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # Log to W&B at regular intervals (every 100 iterations)
        if WANDB_AVAILABLE and wandb.run is not None and args is not None and hasattr(args, 'use_wandb') and args.use_wandb:
            if hasattr(utils, 'get_rank') and utils.get_rank() == 0:
                iteration += 1
                if iteration % 100 == 0:
                    wandb.log({
                        'train/iter_loss': loss_value,
                        'train/iter_lr': optimizer.param_groups[0]["lr"],
                        'train/iter_throughput': img_per_sec,
                    }, commit=False)
    
    # Calculate samples per epoch
    epoch_time = time.time() - epoch_start_time
    samples_per_epoch = total_samples
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Add samples/epoch to the stats
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['samples/epoch'] = samples_per_epoch
    stats['epoch_time'] = epoch_time
    
    print("Averaged stats:", metric_logger)
    if 'img_s' in stats:
        print(f"Throughput: {stats['img_s']:.2f} img/s, Samples per epoch: {samples_per_epoch}")
    
    return stats


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
