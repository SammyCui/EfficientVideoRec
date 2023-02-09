import argparse
import os
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from backbones.resnet import Resnet12
from datasets.VOC import VOCDataset
from foveated_encoder import Foveated_Encoder

DEFAULT_ROOT = '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/'
DEFAULT_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_tuple(x):
    if isinstance(x, tuple):
        return x
    return x, x


def get_model_optimizer(args):
    per_encoder, fov_encoder = Resnet12(), Resnet12()
    model = Foveated_Encoder(per_encoder=per_encoder, fov_encoder=fov_encoder, fov_out_dim=640, per_out_dim=640, per_size=args.per_size, n_classes=10, pe=None)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    else:
        raise Exception("Unknown optimizer")

    if args.lr_scheduler:
        if args.lr_scheduler == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.step_size),
                gamma=args.gamma
            )
        elif args.lr_scheduler == 'multistep':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(_) for _ in args.step_size.split(',')],
                gamma=args.gamma,
            )
        elif args.lr_scheduler == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.max_epoch,
                eta_min=0  # a tuning parameter
            )
        else:
            raise ValueError('Unknown Scheduler')
    else:
        lr_scheduler = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['models'])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['train_epoch']
        args.start_step = checkpoint['train_step']

    return model, optimizer, lr_scheduler


def get_dataloaders(args):
    transform = transforms.Compose([transforms.Resize(args.per_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_dataset = VOCDataset(root=os.path.join(args.train_root, 'root'), anno_root=os.path.join(args.train_root, 'annotations'),
                               cls_to_use=DEFAULT_CLS,
                               transform=transform)
    val_dataset = VOCDataset(root=os.path.join(args.val_root, 'root'), anno_root=os.path.join(args.val_root, 'annotations'),
                               cls_to_use=DEFAULT_CLS,
                               transform=transform)
    test_dataset = VOCDataset(root=os.path.join(args.test_root, 'root'), anno_root=os.path.join(args.test_root, 'annotations'),
                               cls_to_use=DEFAULT_CLS,
                               transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataloader, val_dataloader, test_dataloader


def parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, nargs='?',const=DEFAULT_ROOT, default=DEFAULT_ROOT)
    parser.add_argument('--per_size', type=int, const=64, default=64)
    parser.add_argument('--max_epoch', type=int, const=100, default=100)


    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['multistep', 'step', 'cosine'], nargs='?', const=None)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.2)  # for lr_scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--download', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--save', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--resume', type=str, default=None, nargs='?', const=None)

    args = parser.parse_args()
    args = post_process_args(args)
    return args


def post_process_args(args):
    args.train_root = os.path.join(args.root, 'train')
    args.val_root = os.path.join(args.root, 'val')
    args.test_root = os.path.join(args.root, 'test')
    return args


