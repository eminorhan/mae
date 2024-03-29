import datetime
import os
import sys
import time
import warnings

import presets
import torch
import torch.utils.data
import torchvision
import utils
import numpy as np

from coco_utils import get_coco
from torch import nn
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.models.segmentation import FCN

sys.path.insert(0, os.path.abspath('..'))
import models_vit as vits


### vit pretrained model loading utilities
def load_pretrained_weights(model, pretrained_weights):
    if os.path.isfile(pretrained_weights):    
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


def interpolate_pos_embed(model, checkpoint_model):
    '''
    Interpolate position embeddings for high-resolution. Reference: https://github.com/facebookresearch/deit
    '''
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        layers = [nn.Conv2d(in_channels, channels, 1)]

        super().__init__(*layers)


class vit_fcn_backbone(nn.Module):
    def __init__(self, arch, patch_size, pretrained_weights):
        super().__init__()

        self.patch_size = patch_size
        self.backbone = vits.__dict__[arch + "_patch" + str(patch_size)](num_classes=0, global_pool=False)
        load_pretrained_weights(self.backbone, pretrained_weights)
        self.out_channels = self.backbone.embed_dim

    def forward(self, x):
        o = self.backbone.get_intermediate_layers(x)[0]
        o = torch.permute(o, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        o = o[:, :, 1:]
        w_fmap = x.shape[-2] // self.patch_size
        h_fmap = x.shape[-1] // self.patch_size
        o = o.reshape((o.shape[0], o.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y = {"out": o}
        return y


class vit_fcn_backbone_aux(nn.Module):
    def __init__(self, arch, patch_size, pretrained_weights):
        super().__init__()

        self.patch_size = patch_size
        self.backbone = vits.__dict__[arch + "_patch" + str(patch_size)](num_classes=0, global_pool=False)
        load_pretrained_weights(self.backbone, pretrained_weights)
        self.out_channels = self.backbone.embed_dim

    def forward(self, x):
        l = self.backbone.get_intermediate_layers(x, n=2)
        y = {"out": [], "aux": []}

        o, a = l[-1], l[-2]

        o = torch.permute(o, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        o = o[:, :, 1:]
        w_fmap = x.shape[-2] // self.patch_size
        h_fmap = x.shape[-1] // self.patch_size
        o = o.reshape((o.shape[0], o.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y["out"] = o

        a = torch.permute(a, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        a = a[:, :, 1:]
        a = a.reshape((a.shape[0], a.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y["aux"] = a
        return y


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=520)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 1000, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # load data
    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(True, args))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        sampler=test_sampler, 
        num_workers=args.workers, 
        collate_fn=utils.collate_fn
    )

    print('Number of training images:', len(dataset), 'Number of training iterations per epoch:', len(data_loader), 'Number of classes:', num_classes)
    print('Number of test images:', len(dataset_test), 'Number of test iterations per epoch:', len(data_loader_test))

    # set up model
    full_name = args.arch + "_patch" + str(args.patch_size)
    if full_name in vits.__dict__.keys():
        if args.aux_loss:
            backbone = vit_fcn_backbone_aux(args.arch, args.patch_size, args.pretrained_weights)
        else:
            backbone = vit_fcn_backbone(args.arch, args.patch_size, args.pretrained_weights)
    else:
        print(f"Unknown architecture: {full_name}")
        sys.exit(1)

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    aux_classifier = FCNHead(backbone.out_channels, num_classes) if args.aux_loss else None
    classifier = FCNHead(backbone.out_channels, num_classes)
    model = FCN(backbone, classifier, aux_classifier)

    print(model)

    model.to(device)
    model_without_ddp = model

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.__dict__[args.opt](params_to_optimize, args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        acc_global, acc, iu = confmat.compute()

        # save model and eval results
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth"))
        np.savez(os.path.join(args.output_dir, args.save_prefix + "_evals.npz"), 
            acc_global=acc_global.cpu().numpy(), 
            acc=acc.cpu().numpy(), 
            iu=iu.cpu().numpy(), 
            iu_global=iu.mean().cpu().numpy()
            ) 

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    # basics
    parser.add_argument("--data_path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--aux_loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--batch_size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")

    # logging and saving params
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")
    parser.add_argument("--output_dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--print_freq", default=100, type=int, help="print frequency")

    # model params
    parser.add_argument("--arch", default='', type=str, help='Architecture')
    parser.add_argument("--patch_size", default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--pretrained_weights", default='', type=str, help="Path to pretrained weights to evaluate.")

    # optimizer params
    parser.add_argument("--opt", default="Adam", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")

    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--test_only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--use_deterministic_algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")

    # mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)