from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.dropblock import DropBlock


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 drop_rate: float = 0,
                 drop_block: bool = False,
                 block_size: int = 1,
                 domaxpool: bool = True):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.domaxpool = domaxpool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.domaxpool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked,
                    1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class Resnet12Backbone(nn.Module):

    def __init__(self, block=BasicBlock,
                 dropblock_size: int = 1,
                 embedding_dropout: float = 0,  # dropout for embedding
                 dropblock_dropout: float = 0.1,  # dropout rate for residual layes
                 wider: bool = True,  # True for MetaOptNet, False for TADAM
                 channels: int = 3,
                 domaxpool: bool = True
                 ):
        super().__init__()
        self.inplanes = channels
        if wider:
            num_filters = [64, 160, 320, 640]
        else:
            num_filters = [64, 128, 256, 512]

        if domaxpool:
            domaxpools = [True, True, True, True]
        else:
            domaxpools = [True, True, True, False]
        self.layer1 = self._make_layer(block, num_filters[0], stride=2, dropblock_dropout=dropblock_dropout, domaxpool=domaxpools[0])
        self.layer2 = self._make_layer(block, num_filters[1], stride=2, dropblock_dropout=dropblock_dropout, domaxpool=domaxpools[1])
        self.layer3 = self._make_layer(block, num_filters[2], stride=2, dropblock_dropout=dropblock_dropout,
                                       drop_block=True, block_size=dropblock_size, domaxpool=domaxpools[2])
        self.layer4 = self._make_layer(block, num_filters[3], stride=2, dropblock_dropout=dropblock_dropout,
                                       drop_block=True, block_size=dropblock_size, domaxpool=domaxpools[3])

        self.embedding_dropout = embedding_dropout
        self.dropout = nn.Dropout(p=self.embedding_dropout, inplace=False)
        self.dropblock_dropout = dropblock_dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, stride: int = 1, dropblock_dropout: float = 0, drop_block: bool = False,
                    block_size: int = 1, domaxpool: bool = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dropblock_dropout, drop_block, block_size, domaxpool)]
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = torch.flatten(x, 1)
        return x



def Resnet12(**kwargs):

    return Resnet12Backbone(**kwargs)


if __name__ == '__main__':
    bb = Resnet12Backbone(domaxpool=False)
    inp = torch.zeros((1, 3, 9, 31))
    print(bb(inp).shape)