from typing import Union, Tuple, Any, Callable, Optional
from backbones.resnet import Resnet12Backbone
import torch
import torchvision.transforms
from torch import nn

class Foveated_Encoder(nn.Module):
    def __init__(
        self,
        per_encoder: Callable,
        fov_encoder: Callable,
        pe: Optional[Callable],

        per_out_dim: int,
        fov_out_dim: int,
        n_classes: int
        ):
        super().__init__()

        self.per_encoder = per_encoder
        self.fov_encoder = fov_encoder
        self.per_out_dim = per_out_dim
        self.fov_out_dim = fov_out_dim
        self.fc = nn.Linear(in_features=self.per_out_dim+self.fov_out_dim, out_features=n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.pe = pe

        
    def forward(self, x, fov):
        per_embeddings = self.per_encoder(x)
        per_embeddings = self.avgpool(per_embeddings)
        per_embeddings = torch.flatten(per_embeddings, 1)
        fov_embeddings = self.fov_encoder(fov)
        fov_embeddings = self.avgpool(fov_embeddings)
        if self.pe:
            fov_embeddings = self.pe(fov_embeddings)
        fov_embeddings = torch.flatten(fov_embeddings,1)
        x = torch.cat([fov_embeddings, per_embeddings], dim=1)
        x = self.fc(x)
        return x


class FE_WeightShare(nn.Module):
    def __init__(
            self,
            backbone: Callable,
            pe: Optional[Callable],
            out_dim: int,
            n_classes: int
    ):
        super().__init__()

        self.backbone = backbone
        self.fc = nn.Linear(in_features=out_dim, out_features=n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pe = pe

    def forward(self, x, fov):
        per_embeddings = self.backbone(x)
        per_embeddings = self.avgpool(per_embeddings)
        per_embeddings = torch.flatten(per_embeddings, 1)
        fov_embeddings = self.backbone(fov)
        fov_embeddings = self.avgpool(fov_embeddings)
        if self.pe:
            fov_embeddings = self.pe(fov_embeddings)
        fov_embeddings = torch.flatten(fov_embeddings, 1)
        x = torch.cat([fov_embeddings, per_embeddings], dim=1)
        x = self.fc(x)
        return x




