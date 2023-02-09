from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.nn.common_types import _size_2_t, _size_any_t
from utils.utils import make_tuple

class PerMaxPool2d(nn.Module):
    def __init__(self,
                 pool_size: _size_2_t,
                 center_size: _size_2_t,
                 center_loc: Union[Tuple[int, int], str] = 'center',
                 ):
        """

        :param pool_size: How many 'vertical' pixels to pool
        :param center_size:
        :param center_loc: Either a tuple of (x,y) coordinate of the upper left corner,
                            or one of ['center', 'ur', 'ul', 'br', 'bl']
                                      ('center', 'upper_right', 'upper_left', 'bottom_right', 'bottom_left')
        """
        self.pool_size = pool_size
        self.center_size = make_tuple(center_size)
        self.center_loc = center_loc


    def forward(self, x):
        N, C, H, W = x.shape
        assert self.center_size[0] <= H - self.pool_size * 2 and self.center_size[1] <= W - self.pool_size * 2, \
            'Center size has to be smaller than H - 2 * pool_size, W - 2 * pool_size to be pooled towards center'
        if isinstance(self.center_loc, str):
            if self.center_loc == 'center':
                self.center_loc = ((H-self.center_size[0])//2, (W-self.center_size[1])//2)
            elif self.center_loc == 'ur':
                self.center_loc = (0, W-self.center_size[1])
            elif self.center_loc == 'ul':
                self.center_loc = (0, 0)
            elif self.center_loc == 'br':
                self.center_loc = (H-self.center_size[0], W-self.center_size[1])
            elif self.center_loc == 'bl':
                self.center_loc = (H-self.center_size[0], 0)
        else:


