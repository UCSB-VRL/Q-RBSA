import random
import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st
from scipy import ndimage

import torch
from torchvision import transforms


def input2tuple(val,ndim,default,last_dim_chan=True):
    #import pdb; pdb.set_trace()
    if val is None: out_list = [default,]*ndim
    elif isinstance(val,(list,tuple,np.ndarray)): out_list = list(val)
    else: out_list = [val,]*ndim
    if last_dim_chan: out_list[-1] = default
    return tuple(out_list)


def downsample(X,strides=None,offsets=None,sigmas=None,last_dim_chan=True):
    #import pdb; pdb.set_trace()
    ndim = len(X.shape)
    strides = input2tuple(strides,ndim,1,last_dim_chan)
    offsets = input2tuple(offsets,ndim,0,last_dim_chan)
    sigmas = input2tuple(sigmas,ndim,0,last_dim_chan)

    # blur, set sigmas=0 or sigmas=None to "turn off" blurring
    X_filt = ndimage.gaussian_filter(X,sigmas)

    # subsample the image
    X_ds = X_filt[tuple(slice(o,None,s) for o,s in zip(offsets,strides))]
    return X_ds

def get_patch(img_in, img_hr_in, patch_size, scale):

    ih, iw = img_hr_in.shape[:2]
    tp = patch_size

    ix = random.randrange(0, iw - tp + 1)
    iy = random.randrange(0, ih - tp + 1)

    #img_in = img_in[iy:iy + tp, ix:ix + tp, :]

    img_hr_in = img_hr_in[iy:iy + tp, ix:ix + tp, :]

    # Downsample
    img_ds = downsample(img_hr_in, scale)  

    #return img_ds, img_in
    return img_ds, img_hr_in


def get_patch_1D(img_in, img_hr_in, patch_size, scale):

    ih, iw = img_hr_in.shape[:2]
    tp = patch_size

    ix = random.randrange(0, iw - tp + 1)
    iy = random.randrange(0, ih - tp + 1)

    #img_in = img_in[iy:iy + tp, ix:ix + tp, :]

    img_hr_in = img_hr_in[iy:iy + tp, ix:ix + tp, :]

    # Downsample
    img_ds = img_hr_in[::scale, :, :]
    
    #return img_ds, img_in
    return img_ds, img_hr_in


def get_prog_patch_1D(img_hr_in, epoch, scale):
    #import pdb; pdb.set_trace()
    if epoch <= 200:
        patch_size = 16
    elif epoch > 200 and epoch <=400:
        patch_size = 32
    elif epoch > 400 and epoch <= 800:
        patch_size = 64
    elif epoch > 800:
        patch_size = 100 

    print("Patch Size:", patch_size)   

    bs, ch, ih, iw = img_hr_in.shape
    tp = patch_size

    ix = random.randrange(0, iw - tp + 1)
    iy = random.randrange(0, ih - tp + 1)

    #img_in = img_in[iy:iy + tp, ix:ix + tp, :]

    img_hr_in = img_hr_in[:, :, iy:iy + tp, ix:ix + tp]

    # Downsample
    img_ds = img_hr_in[:, :, ::scale, :]
    
    #return img_ds, img_in
    return img_ds, img_hr_in




def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        #tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]

def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(_l) for _l in l]
