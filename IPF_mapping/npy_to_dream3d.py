import h5py
import numpy as np
import dream3d_import as d3
import glob
import os
from argparser import Argparser

import pdb; pdb.set_trace()
args = Argparser().args

npy_file_dir = f'/data/dkjangid/Material_Projects/superresolution/Quaternion_experiments/saved_weights/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    if args.data == 'Ti64_3D' or args.data == 'Ti64_3D_LR':
        int_part = filename.split("_")[7]
    elif args.data == 'Open_718':
        int_part = filename.split("_")[6]
    else: 
        int_part = filename.split("_")[8]

    return int(int_part) 



file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}/*{args.section}*{args.file_type}*.npy'), key=get_key)

#file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}/*{args.file_type}*{args.section}*.npy'), key=get_key)


total_file = len(file_locs)

arr_list = []
for file_loc in file_locs:
    arr = np.load(file_loc)
    print(arr.shape)
    arr_list.append(arr)

import pdb; pdb.set_trace()
loaded_npy = np.asarray(arr_list)
loaded_npy = np.float32(loaded_npy)

if args.section == 'X_normal' or args.section == 'x_normal':
    loaded_npy = np.moveaxis(loaded_npy, 0, -2)
elif args.section == 'Y_normal' or args.section == 'y_normal':
    loaded_npy = np.moveaxis(loaded_npy, 0, 1)
 
     
d3_sourceName = '/data/dkjangid/superresolution/Material_Dataset/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d'
#d3_sourceName = '/data/dkjangid/superresolution/Material_Dataset/Nickel/raw_data_hdf_files/Open_718_Test.dream3d'

# The path for the output Dream3D file being written.  This is where you want to save the file you are making.

save_path = f'{npy_file_dir}/{args.data}/Dream3D'

if not os.path.exists(f'{npy_file_dir}/{args.data}/Dream3D'):
    os.makedirs(f'{save_path}')

d3_outputName = f'{save_path}/{args.section}_{args.file_type}.dream3d'

d3source = h5py.File(d3_sourceName, 'r')

xdim,ydim,zdim,channeldepth = np.shape(loaded_npy)

phases = np.int32(np.ones((xdim,ydim,zdim)))

new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)


#in_path = 'DataContainers/Test'
in_path = 'DataContainers/ImageDataContainer' 
out_path = 'DataContainers/ImageDataContainer'

new_file = d3.copy_container(d3_sourceName, f'{in_path}/CellEnsembleData', d3_outputName, f'{out_path}/CellEnsembleData')

new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim),
                            source_internal_geometry_path=f'{in_path}/_SIMPL_GEOMETRY',
                            output_internal_geometry_path=f'{out_path}/_SIMPL_GEOMETRY')

new_file = d3.create_empty_container(d3_outputName, f'{out_path}/CellData', (xdim,ydim,zdim), 3)
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', loaded_npy, 'Quats')
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', phases, 'Phases')

# Close out source file to avoid weird memory errors.
d3source.close()
