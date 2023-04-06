#!/bin/sh

python test.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Titanium_all_Z_upsampling' --model 'qrbsa_1d' --n_resblocks 10 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'qrbsa_debug' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only --dist_type 'rotdist'  --scale 4 --syms_type 'HCP'
