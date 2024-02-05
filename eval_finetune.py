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
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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

def get_args_parser():
    parser = argparse.ArgumentParser('MAE finetuning', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int, help='total batch size')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='', type=str, choices=['vit_huge_patch14_476', 'vit_huge_patch14_448', 'vit_huge_patch14', 'vit_large_patch14', 'vit_base_patch14', 'vit_small_patch14'], help='name of model')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)

    # training parameters
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='perform evaluation only')
    parser.add_argument('--no_optim_resume', action='store_true', help='do not resume optim')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--save_prefix", default="", type=str, help='prefix for saving checkpoint and log files')
    parser.add_argument("--frac_retained", default=0.0005, type=float, choices=[0.010147, 0.02, 0.03, 0.05, 0.1, 1.0], help='fraction of train data retained for finetuning')

    return parser

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    cudnn.benchmark = True

    # ============ preparing data ... ============
    # validation transforms
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(args.input_size + 32, interpolation=3),
        pth_transforms.CenterCrop(args.input_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(args.input_size),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16*args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)  # note we use a larger batch size for val

    train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
    # few-shot finetuning
    if args.frac_retained < 1.0:
        print('Fraction of train data retained:', args.frac_retained)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx = indices[:int(args.frac_retained * num_train)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        print(f"Data loaded with {len(train_idx)} train and {len(val_dataset)} val imgs.")
    else:
        print('Using all of train data')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=None)    
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

    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    
    # finetune everything
    for _, p in model.named_parameters():
        p.requires_grad = True
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model_without_ddp = model
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # set optimizer + loss
    loss_scaler = NativeScaler()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        test_stats = evaluate(val_loader, model, device, args)
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_1 = 0.0
    max_accuracy_5 = 0.0
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, loss_scaler, max_norm=None)

        test_stats = evaluate(val_loader, model, device, args.output_dir)
        print(f"Top-1 accuracy of the network on the test images: {test_stats['acc1']:.1f}%")
        print(f"Top-5 accuracy of the network on the test images: {test_stats['acc5']:.1f}%")

        if args.output_dir and test_stats["acc1"] > max_accuracy_1:
            print('Improvement in max test accuracy. Saving model!')
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        max_accuracy_1 = max(max_accuracy_1, test_stats["acc1"])
        max_accuracy_5 = max(max_accuracy_5, test_stats["acc5"])

        print(f'Max accuracy (top-1): {max_accuracy_1:.2f}%')
        print(f'Max accuracy (top-5): {max_accuracy_5:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.save_prefix + "_{}_log.txt".format(args.frac_retained)), mode="a", encoding="utf-8") as f:
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