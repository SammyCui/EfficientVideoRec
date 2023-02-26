from typing import Callable

import torch
from einops import rearrange
from torch import nn
from utils.common_types import _size_2_t
from utils.func_utils import make_tuple
from einops.layers.torch import Rearrange, Reduce
import timm

class BaseReducer(nn.Module):
    def __init__(self, patch_size: _size_2_t, in_chans: int, dim: int, keep_ratio: float, encoder: Callable =None):

        super().__init__()
        self.in_chans = in_chans
        self.patch_size = make_tuple(patch_size)
        self.dim = dim
        #assert self.input_size[0] % self.patch_size == 0, 'input_size should be divisible by patch size'
        self.encoder = encoder if encoder else self._encoder()
        self.score = nn.Softmax(-1)
        self.keep_ratio = keep_ratio


    def _encoder(self):
        # TODO: Could utilize the built-in embedding cnn for vit
        encoder = nn.Sequential(nn.Conv2d(self.in_chans, self.dim, kernel_size=self.patch_size, stride=self.patch_size[0]),
                                nn.Conv2d(self.dim, 1, kernel_size=1, stride=1),
                                Rearrange('b c h w -> b c (h w)'))
        return encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.score(x)
        num2keep = int(self.keep_ratio * x.shape[-1])
        keep_ind = torch.topk(x, num2keep, sorted=False)[1]
        # offset indices by 1 for cls token
        keep_ind = torch.cat((torch.zeros((x.shape[0],1,1)), keep_ind + 1), dim=-1)
        keep_ind = torch.squeeze(keep_ind, dim=1)
        return keep_ind.long()





if __name__ == '__main__':
    vit_tiny = timm.models.vit_tiny_patch16_224()
    inp = torch.rand((1,3,224, 224))
    vit_tiny(inp)

