from model import common
import torch
import torch.nn as nn
from model.quat_utils.Qops_with_QSN import conv2d, Residual_D, First_Residual_D, SNLinear, QSNLinear
from model.quat_utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
import math
from einops import rearrange 

def make_model(args, parent=False):
    return QRCAN_1D(args)

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

        self.conv_layer = conv2d(n_feat, 2*n_feat, kernel_size //2, kernel_size = kernel_size, stride = 1)
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


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                conv2d(channel, channel // reduction,  0 , kernel_size= 1, stride=1),
                nn.ReLU(inplace=True),
                conv2d(channel // reduction, channel, 0, kernel_size = 1, stride = 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv2d(n_feat, n_feat, kernel_size // 2 , kernel_size = kernel_size, stride = 1))
            if bn:
                modules_body.append(QBatchNorm(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        #import pdb; pdb.set_trace()
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                 n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv2d(n_feat, n_feat, kernel_size // 2, kernel_size = kernel_size, stride = 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class QRCAN_1D(nn.Module):
    def __init__(self, args):
        super(QRCAN_1D, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv2d(args.n_colors, n_feats, kernel_size//2 ,kernel_size = kernel_size, stride = 1)]

        # define body module
        modules_body = [
            ResidualGroup(
                 n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv2d(n_feats, n_feats, kernel_size // 2, kernel_size = kernel_size, stride = 1))

        # define tail module
        modules_tail = [
            Upsampler1D(kernel_size, scale, n_feats, act=False),
            conv2d(n_feats, args.n_colors, kernel_size //2, kernel_size = kernel_size, stride = 1)
        ]
 
 
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.epsilon = 0.001


    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
