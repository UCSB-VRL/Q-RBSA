import os
import math
from decimal import Decimal
import utility
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm
from collections import defaultdict
#from plot_cdf import Misorientation_dist 
from mat_sci_torch_quats.quats import fz_reduce, scalar_last2first, scalar_first2last
from mat_sci_torch_quats.symmetries import hcp_syms, fcc_syms  
from collections import defaultdict
import time
import wandb
from thop import profile
import common


class Trainer():
    def __init__(self, args, loader_train, loader_val, loader_test, model, loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp   # checkpoint
        self.loader_train = loader_train
        self.loader_val =  loader_val
        self.loader_test = loader_test
        self.model = model
        self.loss = loss
        self.epoch = args.current_epoch
        self.total_val_loss_all = [0]
        self.epoch_list = [0] 
        self.mis_orient = utility.Misorientation_dist(args)
        self.optimizer = utility.make_optimizer(args, self.model)
        #self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.scheduler = utility.make_warmup_scheduler(args, self.optimizer)


        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8
        self.epsilon = 0.001


    def train(self): 
        #torch.autograd.set_detect_anomaly(True)
        #self.scheduler.step()
        self.loss.step()        
        #epoch = self.scheduler.last_epoch + 1
        self.epoch = self.epoch + 1
        epoch = self.epoch
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        total_train_loss = 0
        
        for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_train):
            #import pdb; pdb.set_trace()
            lr, hr = self.prepare([lr, hr])

            if self.args.prog_patch:
                lr, hr = common.get_prog_patch_1D(hr, epoch, self.args.scale)    
              
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(lr, self.scale)

            _, ch, _, _ = sr.shape

                        
            if isinstance(sr, list):
                loss = np.sum([self.loss(sr[j],hr) for j in range(len(sr))])
            else:
                #import pdb; pdb.set_trace()
                loss = self.loss(sr, hr)

            #if loss.item() > 5000:
            #import pdb; pdb.set_trace()
              
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
            total_train_loss += loss

        #import pdb; pdb.set_trace()
        avg_train_loss = total_train_loss / (batch+1)
        thresh = 1000
        #wandb.log({'train_loss': avg_train_loss})
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
              
        self.scheduler.step()
              
        
    def val_error(self):
        
        #epoch = self.scheduler.last_epoch
        epoch = self.epoch 
        self.ckp.write_log('\nEvaluation:')         
        #import pdb; pdb.set_trace()
        timer_model, timer_data = utility.timer(), utility.timer()
        
        self.model.eval()
                
        with torch.no_grad():
            total_val_loss = 0
            count = 0
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_val):
                eval_acc = 0
             
                #print(f'processing the {filename_lr}')
                lr, hr = self.prepare([lr, hr]) 
                sr = self.model(lr, self.scale)
                
                _, ch, _, _ = sr.shape
 
                org_shape = hr.shape

                if isinstance(sr, list):
                    sr = self.post_process(sr[0], org_shape)
                else:
                    sr = self.post_process(sr, org_shape)

                
                lr = lr.permute(0,2,3,1)
                hr = hr.permute(0,2,3,1)
   
                val_loss = self.mis_orient(sr, hr) 
                val_loss = torch.mean(val_loss)
                val_loss = val_loss.detach().cpu().numpy()
                #lr = lr.detach().cpu().numpy() 
                #hr = hr.detach().cpu().numpy()
                #sr = sr.detach().cpu().numpy()

                #print("val loss:", val_loss)
                total_val_loss += val_loss
                count += 1
        avg_val_loss = total_val_loss / count
        print("avg Val loss:", avg_val_loss)

        #wandb.log({'val_loss': avg_val_loss})        

        #import pdb; pdb.set_trace()             
        self.total_val_loss_all.append(avg_val_loss)
        
        if avg_val_loss <= min(self.total_val_loss_all[1:]): 
           self.ckp.save(self, epoch, is_best = True)  
    
        self.epoch_list.append(epoch)
        self.ckp.plot_val_loss(self.total_val_loss_all, self.epoch_list) 

        if self.args.save_results and (epoch % self.args.save_model_freq) == 0:        
            print("--------------------Saving Model----------------------------")
            self.ckp.save(self,epoch)
        
    
    def test(self, is_trad_results= False):
   
        #import pdb; pdb.set_trace()
        self.model.eval()     
        keys = [f'sr','bilinear', 'bicubic', 'nearest']
        total_psnr_dict = dict.fromkeys(keys,0)
        count = 0
        total_dist = 0
        with torch.no_grad():
            for batch, (lr, hr, filename_lr, filename_hr) in enumerate(self.loader_test):
                start_time = time.time()       
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(f' LR Image: {filename_lr} and HR Image: {filename_hr}')
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                
                modes = []
                sr_up_trad = []
                psnr_dict = defaultdict()
                lr, hr = self.prepare([lr, hr])
                #import pdb; pdb.set_trace() 
                sr = self.model(lr, self.scale)
                
                #sr = hr 
                org_shape = hr.shape
                                               
                #Interpolations
                if is_trad_results:
               
                    modes = ['bilinear', 'bicubic', 'nearest']
                    sr_up_trad = []
                    for mode in modes:
                        upsampling = nn.Upsample(scale_factor=self.scale, mode=mode)
                        sr_up = upsampling(lr)
                        sr_up = self.post_process(sr_up, org_shape)
                        sr_up_trad.append(sr_up)
            
       
                #import pdb; pdb.set_trace() 
                if isinstance(sr, list):
                    sr = self.post_process(sr[0], org_shape)
                else:
                    sr = self.post_process(sr, org_shape)
                
 
                #import pdb; pdb.set_trace() 
                 
                hr = hr.permute(0,2,3,1)
                lr = lr.permute(0,2,3,1)
                             
                 
                save_list = [lr, hr, sr] + sr_up_trad
                #import pdb; pdb.set_trace()
                modes = ['LR', 'HR', f'SR_{self.args.model}_{self.args.model_to_load}_{self.args.dist_type}'] + modes   
                filenames = filename_hr
    
                if self.args.save_results:
                    self.ckp.save_results(filenames, save_list, modes, self.scale, epoch = self.args.model_to_load, dataset=self.args.test_dataset_type) 
                end_time = time.time()
                t = end_time - start_time
                print("Time:", t)

    def post_process(self, x, org_shape):
        #import pdb; pdb.set_trace()
        b, ch, h, w = org_shape
        x = self.normalize(x)
        x = x[:,:,0:h,0:w]
      
         
        x = x.permute(0,2,3,1)
     
        
        # fz_reduction
        x = scalar_last2first(x)

        if self.args.syms_type == 'HCP':      
            x = fz_reduce(x, hcp_syms)
        elif self.args.syms_type == 'FCC':
            x = fz_reduce(x, fcc_syms)        

        x = scalar_first2last(x)
       
        
        return x

    def normalize(self,x):
        x_norm = torch.norm(x, dim=1, keepdim=True)
                # make ||q|| = 1
        y_norm = torch.div(x, x_norm) 

        return y_norm
                                   
    def prepare(self, l, volatile=False):
   
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.cuda()
           
        return [_prepare(_l) for _l in l]

    def upsample(mode, scale):
    
        upsampling = nn.Upsample(scale_factor=scale, mode=mode)
        sr_up = upsampling(lr)
        sr_up = self.normalize(sr_up)

        return sr_up

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            #epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch + 1
            return epoch >= self.args.epochs


    def is_val(self):
        epoch = self.epoch 
        if epoch % self.args.val_freq == 0:
            return True
        else:
            return False   


