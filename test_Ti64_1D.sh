#!/bin/sh

#python test.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Ti64_Bimodal_Unskipped_Z_Upsampling' --model 'qrbsa_1d' --n_resblocks 10 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'qrbsa_layernorm_rotdist_1D_x4_titanium_without_sa' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only --dist_type 'bimodal_unskipped'  --scale 4 --syms_type 'HCP'

python test.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Titanium_all_Z_upsampling' --model 'han_1d' --n_resblocks 20 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'han_rotdist_1D_x2_titanium' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only --dist_type 'rotdist'  --scale 2 --syms_type 'HCP'
