import os
import math
import time
import datetime
import glob
import numpy as np
import scipy.misc as misc
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils import data
#from torchvision import datasets, transforms
import common
from warmup_scheduler import GradualWarmupScheduler
import cv2


class EBSD_Ti64DIC_dataset(data.Dataset):
    """
    Custom Dataset compatible with torch.utils.data.DataLoader
  
    """
    def __init__(self, args, root_lr, root_hr, upsample_2d= True, is_Train=True):
        #import pdb; pdb.set_trace()
        self.args = args
        self.root_lr = root_lr
        self.root_hr = root_hr
        #self.data_lr_files = sorted(glob.glob(f'{root_lr}/*.npy')) if not is_Train else None 
        self.data_lr_files = sorted(glob.glob(f'{root_lr}/*.npy')) if root_lr is not None else None 
        self.data_hr_files = sorted(glob.glob(f'{root_hr}/*.npy'))
        self.is_Train = is_Train
        self.upsample_2D = upsample_2d
        
    def load_file(self, filepath_lr, filepath_hr):
        
        # Load Numpy files
             
        if self.is_Train:
            hr = np.load(f'{filepath_hr}')
            lr = np.load(f'{filepath_lr}') if filepath_lr is not None else None
            #lr, hr = self._get_patch(hr, self.args.patch_size)
            lr, hr = self._get_patch(lr, hr, self.args.patch_size)

        else:
            #hr = np.load(f'{filepath_hr}') 
            #lr, hr = self._get_patch(hr, patch_size=128)

            lr =  np.load(f'{filepath_lr}')
            hr = np.load(f'{filepath_hr}')

        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr, hr = common.np2Tensor([lr, hr], self.args.rgb_range)
        
        filename_hr = os.path.basename(filepath_hr)                                                
        filename_hr = os.path.splitext(filename_hr)[0]                                                                                    
         
        #filename_lr = os.path.basename(filepath_lr)                                                
        #filename_lr = os.path.splitext(filename_lr)[0]                                                                                    
         
        return (lr, hr,filename_hr, filename_hr) 

    def __getitem__(self, index):
     
        #filepath_lr = self.data_lr_files[index] if not self.is_Train else None
        filepath_lr = self.data_lr_files[index] if self.data_lr_files is not None else None 
        filepath_hr = self.data_hr_files[index]

        return self.load_file(filepath_lr, filepath_hr)
 
    def __len__(self):
        return len(self.data_hr_files) 

    def _get_patch(self, lr, hr, patch_size):                                                                                                                                                                                       
        scale = self.args.scale                                                                                                             
        if self.upsample_2D:
            lr, hr = common.get_patch( lr, hr, patch_size, scale)
        else:
            lr, hr = common.get_patch_1D(lr, hr, patch_size, scale) 
                                                                                                                                  
        #lr, hr = common.augment([lr, hr])                                                                                                    
        #lr = common.add_noise(lr, self.args.noise)                                                                                          
                                                                                                                                                                                                                                                                      
        return lr, hr                                                                                                                     

class EBSD_Ti64DIC_Test_dataset(data.Dataset):
    """
    Custom Dataset compatible with torch.utils.data.DataLoader
  
    """
    def __init__(self, args, root_lr, root_hr, is_Train=True):
        #import pdb; pdb.set_trace()
        self.args = args
        self.root_lr = root_lr
        self.root_hr = root_hr
        self.data_hr_files = sorted(glob.glob(f'{root_hr}/*.npy'))
        self.is_Train = is_Train
        
    def load_file(self, filepath_hr):
        
        # Load Numpy files
             
        hr = np.load(f'{filepath_hr}') 
        #lr = common.downsample(hr, self.args.scale)
        hr = hr[0:40,0:40,:]
        lr = hr
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr, hr = common.np2Tensor([lr, hr], self.args.rgb_range)
        filename_hr = os.path.basename(filepath_hr)                                                                                       
        filename_hr = os.path.splitext(filename_hr)[0]                                                                                     

        return (lr, hr,filename_hr, filename_hr) 

    def __getitem__(self, index):  
        filepath_hr = self.data_hr_files[index]

        return self.load_file(filepath_hr)
 
    def __len__(self):
        return len(self.data_hr_files) 

   
    def _get_patch(self, hr):                                                                                                         
        patch_size = self.args.patch_size                                                                                                 
        scale = self.args.scale                                                                                                             
        lr, hr = common.get_patch( hr, patch_size, scale)                                                                                                                             
        #lr, hr = common.augment([lr, hr])                                                                                                    
        #lr = common.add_noise(lr, self.args.noise)                                                                                          
                                                                                                                                                                                                                                                                      
        return lr, hr      

class Misorientation_dist:
    def __init__(self, args, dist_type = 'rot_dist', act = None, syms_req = True):
        
        syms_type = args.syms_type
        print(f'Parameters for Misorientation Distance')
        print('+++++++++++++++++++++++++++++++++++++++++')
        print(f'dist_type: {dist_type}  activation:{act}  symmetry type:{syms_type} Symmetry:{syms_req}') 
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')

        from mat_sci_torch_quats.losses import ActAndLoss, Loss
        from mat_sci_torch_quats.symmetries import hcp_syms, fcc_syms

        if syms_req:
            if syms_type == 'HCP':
                syms = hcp_syms
            elif syms_type == 'FCC':
                syms = fcc_syms
        else:
            syms = None
        
        self.act_loss = ActAndLoss(act, Loss(dist_func=dist_type, syms=syms), quat_dim=-1)
            
    def __call__(self, sr, hr):
        loss = self.act_loss(sr, hr)
        return loss


 
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.':
                args.save = now
            self.dir = '/data/dkjangid/Material_Projects/superresolution/Quaternion_experiments/saved_weights/' + args.save
        else:
            self.dir = '/data/dkjangid/Material_Projects/superresolution/Quaternion_experiments/saved_weights/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        #self.plot_psnr(epoch)
        #torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def plot_val_loss(self, total_val_loss_all, epoch_list):
        #axis = np.linspace(0, epoch, epoch+1)
        axis = epoch_list
        label = 'Misorientation Loss on Validation Data'
        fig = plt.figure()
        plt.title(label)
        
        plt.plot(
                axis, 
                total_val_loss_all,
                label = label
                )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MisOrientation Error')
        plt.grid(True)
        plt.savefig('{}/val_loss.pdf'.format(self.dir))
        plt.close(fig)


        

    def save_results(self, filenames, save_list, postfix, scale, epoch, dataset='val'):
        
        results_dir = f'{self.dir}/results'
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        kwargs_imshow = {'vmin': -1, 'vmax': 1}

        if not self.args.scalar_first:
            channels = ['q1', 'q2', 'q3', 'q0']
        else:
            channels = ['q0', 'q1', 'q2', 'q3']

        #import pdb; pdb.set_trace()
        for idx, file_name in enumerate(filenames):
            #base_name = os.path.basename(file_path)
            #file_name = os.path.splitext(base_name)[0][:-3]                        
            
            if not os.path.exists(f'{self.dir}/results/{dataset}_{epoch}'):
                os.mkdir(f'{self.dir}/results/{dataset}_{epoch}')

            filename = '{}/results/{}_{}/{}_x{}'.format(self.dir, dataset, epoch, file_name, scale)
        
            if self.args.n_colors !=3: 
                for ch_num, channel in enumerate(channels):
                    fig, axes = plt.subplots((len(postfix)+1)//3,3, figsize=(14,12), constrained_layout = True)
                    #fig.tight_layout(pad=2.0)
                    fig.suptitle(f'{dataset} data: Filename:{file_name}_{channel}' , fontweight ="bold")

                    for a, img, title in zip(axes.reshape(-1), save_list, postfix):
                        img_arr = img[idx].cpu().numpy()
                        if ch_num == 0: # save only one time 
                            np.save(f'{filename}_{title}.npy', img_arr)
                        img_numpy = img_arr[:,:, ch_num]
                        im = a.imshow(img_numpy, **kwargs_imshow, cmap='jet')
                        a.set_title(title, fontweight="bold")

                    cbar =fig.colorbar(im, ax = axes.ravel().tolist(), shrink=0.95)
                    cbar.set_ticklabels(np.arange(0,1,0.2))
                    cbar.set_ticklabels([-1 , 0, 1])
                
                    plt.savefig(f'{filename}_{channel}.png')
                
                    plt.close()
            else:
                img_numpy = (np.transpose(img_arr, (1,2,0)) + 1 / 2.0) * 255.0
                img_numpy = np.clip(img_numpy, 0, 255).astype(np.uint8)
                #import pdb; pdb.set_trace()
                #fig.delaxes(axes[2,1]) 
                                
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)



def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calc_psnr_quat(dist, rgb_range):
    dist = dist / rgb_range
    #rmse = (dist**2).mean().sqrt()
    #ps = 20*torch.log10(1/rmse)
    ps = 20*torch.log10(1/dist)

    return ps


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_warmup_scheduler(args, my_optimizer):
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(my_optimizer, args.epochs-warmup_epochs, eta_min=args.optim_lr_min)
    scheduler = GradualWarmupScheduler(my_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    return scheduler

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler

