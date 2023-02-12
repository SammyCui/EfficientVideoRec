from typing import Callable

import torch
from torch import nn


class Benchmark(nn.Module):
    def __init__(self,
                 backbone: Callable,
                 out_dim: int,
                 n_classes: int = 10,
                 ):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features=out_dim, out_features=n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x, fov=None):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x