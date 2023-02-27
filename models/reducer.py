from typing import Callable

import torch
from einops import rearrange
from torch import nn
from utils.common_types import _size_2_t
from utils.func_utils import make_tuple
from einops.layers.torch import Rearrange, Reduce
import timm

class BaseReducer(nn.Module):
    def __init__(self, patch_size: _size_2_t, in_chans: int = 3, dim: int = 32, keep_ratio: float = 0.8, encoder: Callable =None):

        super().__init__()
        self.in_chans = in_chans
        self.patch_size = make_tuple(patch_size)
        self.dim = dim
        assert keep_ratio <= 1, "Keep Ratio has to be <= 1"
        #assert self.input_size[0] % self.patch_size == 0, 'input_size should be divisible by patch size'
        self.encoder = encoder if encoder else self._encoder()
        self.keep_ratio = keep_ratio



    def _encoder(self):
        # TODO: Could utilize the built-in embedding cnn for vit
        encoder = nn.Sequential(nn.Conv2d(self.in_chans, self.dim, kernel_size=self.patch_size, stride=self.patch_size[0]),
                                nn.Conv2d(self.dim, 1, kernel_size=1, stride=1),
                                Rearrange('b c h w -> b c (h w)'),
                                nn.Softmax(-1))
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        num2keep = int(self.keep_ratio * x.shape[-1])
        keep_ind = torch.topk(x, num2keep, sorted=False)[1]

        keep_ind = torch.squeeze(keep_ind, dim=1)
        return keep_ind.long()


class RandomReducer(BaseReducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _encoder(self):
        return nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        N = H // self.patch_size[0] * W // self.patch_size[1]
        num2keep = int(self.keep_ratio * N)
        uniform_weights = torch.ones(N, device=x.device).expand(B, -1)
        keep_ind = torch.multinomial(uniform_weights, num_samples=num2keep, replacement=False)
        return keep_ind.long()


if __name__ == '__main__':
    red = BaseReducer(patch_size=16, in_chans=3)
    img = torch.rand(2,3,224,224)
    red(img)
    print()

