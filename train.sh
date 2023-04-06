#!/bin/sh

python main.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Titanium_all_Z_upsampling' --hr_data_dir 'Train/HR_Images/preprocessed_imgs_1D' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_1D'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_1D' --model 'qrbsa_1d' --n_resblocks 10 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'qrbsa_debug'  --loss '1*MisOrientation' --dist_type 'rot_dist_approx' --patch_size 128 --batch_size 4 --scale 4 --val_freq 1 --save_model_freq 1 --syms_type 'HCP' --syms_req --prog_patch  

