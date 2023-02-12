import argparse
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from backbones.resnet import resnet12, resnet18
from datasets.VOC import VOCDataset
from models.foveated_encoder import *
from models.benchmark import Benchmark

DEFAULT_ROOT = '/u/erdos/students/xcui32/SequentialTraining/datasets/VOC2012/VOC2012_filtered/'
DEFAULT_CLS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow']
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_tuple(x):
    if isinstance(x, tuple):
        return x
    return x, x


def get_model_optimizer(args):

    backbone = eval(args.backbone)(args)
    if args.model == 'Benchmark':
        model = Benchmark(backbone, out_dim=args.backbone_out_dim)
    elif args.model == 'FE_WeightShare':
        model = FE_WeightShare(backbone, out_dim=args.backbone_out_dim, n_classes=10, pe=args.pe)
    else:
        raise Exception('Model not implemented')

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
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    train_dataset = VOCDataset(root=os.path.join(args.train_root, 'root'), anno_root=os.path.join(args.train_root, 'annotations'),
                               cls_to_use=DEFAULT_CLS,
                               transform=transform,
                               per_size=args.per_size)
    val_dataset = VOCDataset(root=os.path.join(args.val_root, 'root'), anno_root=os.path.join(args.val_root, 'annotations'),
                            cls_to_use=DEFAULT_CLS,
                            transform=transform,
                            per_size=args.per_size)
    test_dataset = VOCDataset(root=os.path.join(args.test_root, 'root'), anno_root=os.path.join(args.test_root, 'annotations'),
                              cls_to_use=DEFAULT_CLS,
                              transform=transform,
                              per_size=args.per_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader



def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, nargs='?', const=DEFAULT_ROOT, default=DEFAULT_ROOT)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)

    parser.add_argument('--backbone', type=str, nargs='?', const='resnet18', default='resnet18')
    parser.add_argument('--model', type=str, nargs='?', const='Benchmark', default='Benchmark')
    parser.add_argument('--backbone_out_dim', type=int, nargs='?', const=512, default=512)
    parser.add_argument('--pe', type=str, nargs='?', const=None,default=None)
    parser.add_argument('--per_size', type=int, nargs='?', const=None, default=None)
    parser.add_argument('--base_channels', type=int, nargs='?', const=64, default=64)

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

    args.train_root = os.path.join(args.root, 'train')
    args.val_root = os.path.join(args.root, 'val')
    args.test_root = os.path.join(args.root, 'test')
    args.pretrained = eval(args.pretrained)
    args.num_classes = len(DEFAULT_CLS)
    if args.device == 'gpu':
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return args


class DebugArgs:
    def __init__(self,

                 root: str = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered',
                 backbone: str = 'resnet18',
                 model: str = 'Benchmark',
                 backbone_out_dim: int = 512,
                 pe: str = None,
                 per_size: int = None,
                 base_channels: int = 64,
                 start_epoch: int = 0,
                 max_epoch: int = 200,
                 lr: float = 0.001,
                 optimizer: str = 'adam',
                 lr_scheduler: str = 'step',
                 step_size: int = 20,
                 num_classes: int = 10,
                 gamma: float = 0.2,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 val_interval: int = 1,
                 num_workers: int = 1,
                 batch_size: int = 2,
                 pretrained: bool = False,
                 download: bool = False,
                 device: str = 'cpu',
                 result_dir: str = './checkpoints',
                 save: bool = False,
                 resume: bool = False,
                 init_backbone: bool = False):

        self.root = root
        self.backbone = backbone
        self.backbone_out_dim = backbone_out_dim
        self.pe = pe
        self.per_size = per_size
        self.base_channels = base_channels
        self.model = model
        self.download = download
        self.num_workers = num_workers
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.num_classes = num_classes
        self.lr = lr
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
