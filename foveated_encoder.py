from typing import Union, Tuple, Any, Callable, Optional

import torch
import torchvision.transforms
from torch import nn
from torch.nn.common_types import _size_2_t

class Foveated_Encoder(nn.Module):
    def __init__(
        self,
        per_encoder: Callable,
        fov_encoder: Callable,
        pe: Optional[Callable],
        per_size: _size_2_t,
        per_out_dim: int,
        fov_out_dim: int,
        n_classes: int,
        device=None,
        ):
        super().__init__()

        self.per_encoder = per_encoder
        self.fov_encoder = fov_encoder
        self.per_size = per_size
        self.per_out_dim = per_out_dim
        self.fov_out_dim = fov_out_dim
        self.device = device
        self.fc = nn.Linear(in_features=self.per_out_dim+self.fov_out_dim, out_features=n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.pe = pe

        
    def forward(self, x, bb):
        x_min, y_min, x_max, y_max = bb
        bb = x[x_min:x_max, y_min:y_max]
        x = torchvision.transforms.Resize(self.per_size)(x)
        per_embeddings = self.per_encoder(x)
        per_embeddings = self.avgpool(per_embeddings)
        per_embeddings = torch.flatten(per_embeddings, 1)
        fov_embeddings = self.fov_encoder(bb)
        fov_embeddings = self.avgpool(fov_embeddings)
        if self.pe:
            fov_embeddings = self.pe(fov_embeddings)
        fov_embeddings = torch.flatten(fov_embeddings,1)
        x = fov_embeddings + per_embeddings
        x = self.fc(x)
        return x
        