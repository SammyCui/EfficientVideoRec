from typing import Callable

import torch
from einops import rearrange
from torch import nn
from utils.common_types import _size_2_t
from utils.func_utils import make_tuple
from einops.layers.torch import Rearrange, Reduce
import timm

class BaseReducer(nn.Module):
    def __init__(self, patch_size: _size_2_t, in_chans: int = 3, dim: int = 32, keep_ratio: float = 0.8, encoder: Callable =None, **kwargs):

        super().__init__()
        self.in_chans = in_chans
        self.patch_size = make_tuple(patch_size)
        self.dim = dim
        assert keep_ratio <= 1, "Keep Ratio has to be <= 1"
        #assert self.input_size[0] % self.patch_size == 0, 'input_size should be divisible by patch size'
        self.encoder = encoder if encoder else \
            nn.Sequential(nn.Conv2d(self.in_chans, self.dim, kernel_size=self.patch_size, stride=self.patch_size[0]),
                                nn.BatchNorm2d(self.dim),
                                nn.ReLU(),
                                nn.Conv2d(self.dim, 1, kernel_size=1, stride=1),
                                Rearrange('b c h w -> b c (h w)'))
        self.keep_ratio = keep_ratio

    def forward(self, x):
        x = self.encoder(x)
        num2keep = int(self.keep_ratio * x.shape[-1])
        keep_ind = torch.topk(x, num2keep, sorted=False)[1]

        keep_ind = torch.squeeze(keep_ind, dim=1)
        return keep_ind.long()


class RandomReducer(BaseReducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H // self.patch_size[0] * W // self.patch_size[1]
        num2keep = int(self.keep_ratio * N)
        uniform_weights = torch.ones(N, device=x.device).expand(B, -1)
        keep_ind = torch.multinomial(uniform_weights, num_samples=num2keep, replacement=False)
        return keep_ind.long()


class conv_block(nn.Module):
    def __init__(self, in_chans, out_chans, downsample=False, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
        )
        if downsample:
            self.conv.add_module('MaxPool2d', nn.MaxPool2d(2))

    def forward(self, x):
        return self.conv(x)


class ConvReducer(BaseReducer):
    def __init__(self, reducer_depth, image_size = 224, **kwargs):
        super().__init__(**kwargs)
        self.image_size = make_tuple(image_size)
        self.down_chans = [self.in_chans] + [self.dim * 2 ** i for i in range(reducer_depth)]
        self.spatial_conv = nn.ModuleList([conv_block(self.down_chans[i], self.down_chans[i + 1], downsample=True, padding=1)
                                           for i in range(reducer_depth)])
        p_h, p_w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        self.avgpool = nn.AdaptiveAvgPool2d((p_h, p_w))
        self.channel_conv = nn.Conv2d(self.down_chans[-1], 1, kernel_size=1)


    def forward(self, x):

        for conv in self.spatial_conv:
            x = conv(x)

        x = self.avgpool(x)
        x = self.channel_conv(x)
        x = rearrange(x, 'b c h w -> b c (h w)')
        num2keep = int(self.keep_ratio * x.shape[-1])
        keep_ind = torch.topk(x, num2keep, sorted=False)[1]

        keep_ind = torch.squeeze(keep_ind, dim=1)
        return keep_ind.long()






if __name__ == '__main__':
    red = ConvReducer(depth=3, patch_size=16, image_size=224)
    inp = torch.rand(1,3,64,64)
    out = red(inp)
    print(out.shape)

