import torch
from torch.utils.data import DataLoader
#from torch import optim
import utility
import model
import loss
from utility import  EBSD_Ti64DIC_Test_dataset, EBSD_Ti64DIC_dataset

from argparser import Argparser
import os
from trainer import Trainer

args = Argparser().args
checkpoint = utility.checkpoint(args)


if checkpoint.ok:
   
    """
    Test Data Loader

    """
   
    import pdb; pdb.set_trace()
    lr_data_path = f'/{args.input_dir}/{args.test_dataset_type}/LR_Images/X{args.scale}/preprocessed_imgs_1D'
    hr_data_path = f'/{args.input_dir}/{args.test_dataset_type}/HR_Images/preprocessed_imgs_1D'
    #lr_data_path = hr_data_path 

    print(f'LR {args.test_dataset_type} Path: {lr_data_path}')
    print(f'HR {args.test_dataset_type} Path: {hr_data_path}')
   
    #lr_data_path = None
    #dataset = EBSD_Ti64DIC_Test_dataset(args, lr_data_path, hr_data_path, is_Train=False) 
    dataset = EBSD_Ti64DIC_dataset(args, lr_data_path, hr_data_path, is_Train=False)  
    data_loader_test = DataLoader(dataset=dataset, batch_size=args.val_batch_size, 
                             num_workers= 1, 
                             shuffle=False, drop_last=False)

        
    model = model.Model(args, checkpoint)
        
    loss = loss.Loss(args, checkpoint) if not args.test_only else None 
    
    data_loader_train = None
    data_loader_val = None

    t = Trainer(args, data_loader_train, data_loader_val, data_loader_test, model, loss, checkpoint) 
        
    t.test(is_trad_results=False)


