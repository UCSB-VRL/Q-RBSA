import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        #import pdb; pdb.set_trace()
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        #import pdb; pdb.set_trace()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                #m.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                #m.append(conv(n_feat, n_feat, 3, bias))

                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                #m.append(conv(n_feat, 2 * n_feat, 3, bias))
                # m.append(conv(2 * n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        #import pdb; pdb.set_trace()
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        long_height = x.shape[2]
        short_width = x.shape[3]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, long_height, short_width])
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_height, long_width)

        return x

class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x



class Upsampler1D(nn.Module):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        super(Upsampler1D, self).__init__()

        self.conv_layer = conv(n_feat, 2*n_feat, 3, bias)
        self.pixel_shuffle = PixelShuffle1D(2) 
        #self.up_sample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
      
        self.scale = scale
        self.n_feat = n_feat
         
    def forward(self, x):
        x = x.permute(0,1,3,2)
        if (self.scale & (self.scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(self.scale, 2))):
                #import pdb; pdb.set_trace()
                bsize, ch, h, w = x.shape
                x = self.conv_layer(x)
                #x = rearrange(x, 'd0 d1 d2 d3 -> d0 d1 (d2 d3)')
                x = self.pixel_shuffle(x)
                #x = rearrange(x, 'd0 d1 (d2 d3) -> d0 d1 d2 d3', d0=bsize, d1=ch, d2=h, d3=2*w)
                
            x = x.permute(0,1,3,2) 
            return x
                      
        else:
            raise NotImplementedError


