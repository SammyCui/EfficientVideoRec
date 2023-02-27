import argparse
import os
from typing import List
import re
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models.vit import vit_benchmark, reducer_vit

from backbones.resnet import resnet12, resnet18

from datasets.VOC import VOCDataset
from models.foveated_encoder import *
from models.benchmark import Benchmark

DEFAULT_ROOT = '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/'
DEFAULT_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)





def get_model_optimizer(args):

    if args.mode == '2streamCNN':
        backbone = eval(args.backbone)(args)
        if args.model == 'Benchmark':
            model = Benchmark(backbone, out_dim=args.backbone_out_dim)
        elif args.model == 'FE_WeightShare':
            model = FE_WeightShare(backbone, out_dim=args.backbone_out_dim, n_classes=10, pe=args.pe, concat=args.concat)
        else:
            raise Exception('Model not implemented')

    elif args.mode == 'reducer-img':
        model = eval(args.model)(args)

    else:
        raise Exception('Mode not defined')

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
    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
    # val_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)
    train_dataset = VOCDataset(root=os.path.join(args.train_root, 'root'), anno_root=os.path.join(args.train_root, 'annotations'),
                               cls_to_use=args.cls_to_use,
                               transform=transform,
                               per_size=args.per_size,
                               object_only=args.object_only)
    val_dataset = VOCDataset(root=os.path.join(args.val_root, 'root'), anno_root=os.path.join(args.val_root, 'annotations'),
                            cls_to_use=args.cls_to_use,
                            transform=transform,
                            per_size=args.per_size,
                            object_only=args.object_only)
    test_dataset = VOCDataset(root=os.path.join(args.test_root, 'root'), anno_root=os.path.join(args.test_root, 'annotations'),
                              cls_to_use=args.cls_to_use,
                              transform=transform,
                              per_size=args.per_size,
                              object_only=args.object_only)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader



def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, nargs='?', const=DEFAULT_ROOT, default=DEFAULT_ROOT)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train', type=str, default='True')
    parser.add_argument('--mode', type=str, default='reducer-img')
    parser.add_argument('--model', type=str, nargs='?', const='Benchmark', default='Benchmark')
    parser.add_argument('--model_variant', type=str)
    parser.add_argument('--reducer', type=str, default='RandomReducer')
    parser.add_argument('--patch_size', type=int, default=16)

    parser.add_argument('--object_only', type=str, default='False')
    parser.add_argument('--subset_data', type=str, default='False')
    parser.add_argument('--reducer_inner_dim', type=int, default=32)
    parser.add_argument('--keep_ratio', type=float, default=0.8)
    parser.add_argument('--backbone', type=str, nargs='?', const='resnet18', default='resnet18')
    parser.add_argument('--backbone_out_dim', type=int, nargs='?', const=512, default=512)
    parser.add_argument('--pe', type=str, nargs='?', const=None,default=None)
    parser.add_argument('--per_size', type=int, nargs='?', const=None, default=None)
    parser.add_argument('--base_channels', type=int, nargs='?', const=64, default=64)
    parser.add_argument('--concat', type=str, default='False')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=['multistep', 'step', 'cosine'], nargs='?', const=None)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.2)  # for lr_scheduler
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--download', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--save', type=str, default='False', nargs='?', const='False')
    parser.add_argument('--resume', type=str, default=None, nargs='?', const=None)
    parser.add_argument('--pretrained', type=str, default='False')
    parser.add_argument('--init_backbone', type=str, default=None, nargs='?', const=None)

    args = parser.parse_args()
    args = post_process_args(args)
    return args


def post_process_args(args):

    if args.patch_size and args.model_variant and args.pretrained:
        reg = re.compile("patch(\d+)")
        pretrained_patchsize = int(reg.findall(args.model_variant)[0])
        assert pretrained_patchsize == args.patch_size, 'Specified patch size has to be equal to the pretrained patch size'

    args.train_root = os.path.join(args.root, 'train')
    args.val_root = os.path.join(args.root, 'val')
    args.test_root = os.path.join(args.root, 'test')
    args.pretrained = eval(args.pretrained)
    args.concat = eval(args.concat)
    args.train = eval(args.train)
    args.object_only = eval(args.object_only)
    args.subset_data = eval(args.subset_data)
    # if subset data, take 10 default classes
    if args.subset_data:
        args.num_classes = 10
        args.cls_to_use = DEFAULT_CLS
    else:
        args.num_classes = 20
        args.cls_to_use = None
    if args.device == 'gpu':
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


class DebugArgs:
    def __init__(self,

                 root: str = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered',
                 backbone: str = 'resnet18',
                 model: str = 'FE_WeightShare',
                 mode: str = 'reducer-img',
                 keep_ratio: float = .8,
                 reducer_inner_dim: int = 64,
                 patch_size: int = 16,
                 backbone_out_dim: int = 512,
                 model_variant: str = 'vit_tiny_patch16_224',
                 pe: str = None,
                 per_size: int = None,
                 concat: bool = False,
                 base_channels: int = 64,
                 start_epoch: int = 0,
                 reducer: str = 'BaseReducer',
                 object_only: bool = False,
                 max_epoch: int = 200,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 lr_scheduler: str = 'step',
                 step_size: int = 20,
                 gamma: float = 0.2,
                 num_classes: int =20,
                 cls_to_use: List[str] = None,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 train: bool = True,
                 val_interval: int = 1,
                 num_workers: int = 1,
                 batch_size: int = 2,
                 pretrained: bool = True,
                 download: bool = False,
                 device: str = 'cpu',
                 result_dir: str = './results',
                 save: bool = False,
                 resume: bool = False,
                 init_backbone: bool = False):

        self.root = root
        self.backbone = backbone
        self.mode = mode
        self.cls_to_use = cls_to_use
        self.keep_ratio = keep_ratio
        self.patch_size = patch_size
        self.model_variant = model_variant
        self.backbone_out_dim = backbone_out_dim
        self.pe = pe
        self.per_size = per_size
        self.base_channels = base_channels
        self.model = model
        self.reducer = reducer
        self.reducer_inner_dim = reducer_inner_dim
        self.object_only = object_only
        self.download = download
        self.concat = concat
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.lr = lr
        self.train = train
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step_size = step_size
        self.gamma = gamma
        self.momentum = momentum
        self.pretrained = pretrained
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.device = device
        self.result_dir = result_dir
        self.save = save
        self.resume = resume
        self.init_backbone = init_backbone

        self.train_root = os.path.join(self.root, 'train')
        self.val_root = os.path.join(self.root, 'val')
        self.test_root = os.path.join(self.root, 'test')

