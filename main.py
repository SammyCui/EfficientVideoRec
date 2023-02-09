import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from timeit import default_timer as timer
import os
import argparse
from backbones.resnet import Resnet12
from foveated_encoder import Foveated_Encoder
from datasets.VOC import VOCDataset
from utils.meters import AverageMeter
from utils.metric import accuracy



if __name__ == '__main__':
    args = parser()



    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.step_size),
        gamma=args.gamma
    )

    train_time, forward_tm, backward_tm, optimize_tm = (AverageMeter(),) * 4

    for epoch in range(args.epoch):
        epoch_t0 = timer()
        model.train()
        train_acc, train_loss = AverageMeter(), AverageMeter()
        val_acc, val_loss = AverageMeter(), AverageMeter()
        for data, bb, labels in train_dataloader:
            data, bb, labels = data.to(device), bb.to(device), labels.to(device)
            optimizer.zero_grad()
            forward_t0 = timer()
            outputs = model(data)
            forward_t1 = timer()
            loss = criterion(outputs, labels)

            backward_t0 = timer()
            loss.backward()
            backward_t1 = timer()

            optimizer_t0 = timer()
            optimizer.step()
            optimizer_t1 = timer()

            cur_acc = accuracy(outputs, labels)
            train_acc.update(cur_acc[0].item())
            train_loss.update(loss.item())

            backward_tm.update(backward_t1 - backward_t0)
            forward_tm.update(forward_t1 - forward_t0)
            optimize_tm.update(optimizer_t1 - optimizer_t0)
            epoch_t1 = timer()
            train_time.update(epoch_t1-epoch_t0)
            lr_scheduler.step()
        with torch.no_grad():
            model.eval()
            for data, labels in val_dataloader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                val_loss.update(criterion(outputs, labels).item())
                val_acc.update(accuracy(outputs, labels)[0].item())

        print(
            f'Epoch: {epoch}, loss: {train_loss.avg}, acc: {train_acc.avg}, val_loss: {val_loss.avg}, val_acc: {val_acc.avg}')

