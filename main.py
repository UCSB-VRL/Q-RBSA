import torch
from torch.utils.data import DataLoader
import utility
import model
import loss
import os
import numpy as np
import random

from utility import EBSD_Ti64DIC_dataset
from argparser import Argparser
from trainer import Trainer
import wandb
from prettytable import PrettyTable

args = Argparser().args
#wandb.init(project="EBSDSR_Z_Upsampling_Networks_X2", config=args)


checkpoint = utility.checkpoint(args)

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if checkpoint.ok:
   
    
    """
    Train and Validation Data Loader

    """

    import pdb; pdb.set_trace()
    #lr_train_data_path = f'/{args.input_dir}/{args.lr_data_dir}'
    lr_train_data_path = None
    hr_train_data_path = f'/{args.input_dir}/{args.hr_data_dir}' 

    print("LR Train Path:", lr_train_data_path)
    print("HR Train Path:", hr_train_data_path)
     
    lr_val_data_path = f'/{args.input_dir}/{args.val_lr_data_dir}' 
    hr_val_data_path = f'/{args.input_dir}/{args.val_hr_data_dir}' 

    print("LR Val Path:", lr_val_data_path)
    print("HR Val Path:", hr_val_data_path)
     

    dataset_train = EBSD_Ti64DIC_dataset(args, lr_train_data_path, hr_train_data_path, upsample_2d=args.upsample_2d) 
    dataset_val = EBSD_Ti64DIC_dataset(args, lr_val_data_path, hr_val_data_path, is_Train=False) 
 


    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, 
                             num_workers= 16, 
                             shuffle=True, drop_last=True)       
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.val_batch_size, 
                             num_workers= 1, 
                             shuffle=False, drop_last=False)

    data_loader_test = None

    model = model.Model(args, checkpoint)
    #import pdb; pdb.set_trace()
    count_parameters(model)
            
    loss = loss.Loss(args, checkpoint) if not args.test_only else None 
    t = Trainer(args, data_loader_train, data_loader_val, data_loader_test,  model, loss, checkpoint) 
        
    while not t.terminate():
        t.train()
        
        if t.is_val():
            t.val_error()


