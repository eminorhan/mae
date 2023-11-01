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
import webdataset as wds

import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchvision import transforms as pth_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import models_vit

from engine_finetune import train_one_epoch, evaluate


def identity(x):
    return x

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int, help='total batch size')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, choices=['vit_huge_patch14', 'vit_large_patch16', 'vit_base_patch16', 'vit_small_patch16'], help='Name of model to train')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--split', default=False, action='store_true', help='whether to manually split dataset into train-val')
    parser.add_argument('--subsample', default=False, action='store_true', help='whether to subsample the data')

    # training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # validation transforms
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.split:
        from torch.utils.data.sampler import SubsetRandomSampler

        val_dataset = ImageFolder(args.train_data_path, transform=val_transform)
        train_dataset = ImageFolder(args.train_data_path, transform=train_transform)

        num_train = len(train_dataset)
        print('Total data size is', num_train)

        indices = list(range(num_train))
        np.random.shuffle(indices)

        if args.subsample:
            num_data = int(0.1 * num_train)
            train_idx, test_idx = indices[:(num_data // 2)], indices[(num_data // 2):num_data]
        else:
            split = int(np.floor(0.5 * num_train))  # split 50-50, change here if you need to do sth else
            train_idx, test_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

        print(f"Data loaded with {len(train_idx)} train and {len(test_idx)} val imgs.")
        print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")
    else:
        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
        print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")
    # ============ done data ... ============
    
    # set up and load model
    model = models_vit.__dict__[args.model](num_classes=args.num_labels, global_pool=args.global_pool)

    if args.resume and not args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.resume)
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

    # set optimizer + loss
    loss_scaler = NativeScaler()
    optimizer = torch.optim.Adam(model_without_ddp.head.parameters(), args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # load if resuming from a checkpoint; I need to update the above resume probably
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(val_loader, model, device, args)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler, max_norm=None)

        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(val_loader, model, device, args.output_dir)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.save_prefix + "_log.txt"), mode="a", encoding="utf-8") as f:
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