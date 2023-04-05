import torch
import torch.nn as nn
import math

from model.quat_utils.Qops_with_QSN import conv2d, Residual_D, First_Residual_D, SNLinear, QSNLinear
from model.quat_utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
from model import common
from einops import rearrange 


def make_model(args, parent=False):
    return QEDSR(args)


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
    def __init__(self, kernel_size, scale, n_feat, bn=False, act=False, bias=True):
        super(Upsampler1D, self).__init__()

        self.conv_layer = conv2d(n_feat, 2*n_feat, kernel_size = kernel_size, stride = 1, padding= kernel_size //2)
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


class QEDSR(nn.Module):
    def __init__(self, args, spectral_normed=False, channel=4):
        super(QEDSR, self).__init__()

        #import pdb; pdb.set_trace()
        self.spectral_normed = spectral_normed
        # self.batch_normed = batch_normed
       
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)
 
        m_head = [conv2d(args.n_colors, n_feats,  kernel_size = kernel_size, stride = 1, padding=kernel_size//2)]

        m_body = [Residual_D(n_feats, n_feats, spectral_normed = spectral_normed, down_sampling = False)  for _ in range(n_resblocks)]

        m_body.append(conv2d(n_feats, n_feats, kernel_size = kernel_size, stride = 1, padding=kernel_size//2))

        m_tail = [
            Upsampler1D(kernel_size, scale, n_feats, act=False),
            conv2d(n_feats, args.n_colors, kernel_size = kernel_size, stride = 1, padding=kernel_size//2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
       
    def forward(self, x):

        #import pdb; pdb.set_trace()
        x = self.head(x)

        res = self.body(x)
        res += x
        
        x = self.tail(res) 
        
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

