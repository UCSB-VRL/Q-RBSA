import numpy as np
import glob
import os
import imageio
import pdb; pdb.set_trace()

data = 'Test'
loss_type = 'rotdist'
material='Ti64_Bimodal_Unskipped'
network= 'qrbsa_layernorm'
model_name=f'{network}_{loss_type}_1D_x4_titanium_progpatch'
datatype = 'HR'

#if datatype == 'HR':
#    fdir = f'/data2/dkjangid/Material_Projects/superresolution/AAAI22/saved_weights/HR_1D_IPF_Test_newpreprocess/{material}'   
#else:
fdir = f'/data/dkjangid/Material_Projects/superresolution/Quaternion_experiments/saved_weights/{model_name}/results/{data}_model_best/{material}/Dream3D'

#planes = ('X_normal', 'Y_normal')
planes = ('x_normal', 'y_normal')


sizes = {'Ti64_3D': {
                'Test': [120, 142, 270, 3],
                'Val': [120, 142, 200, 3]
                },
        'Ti7_1Percent_3D': {
                'Test': [232, 262, 202, 3],
                'Val': [232, 195, 202, 3]
                },
        'Ti7_3Percent_3D': {
                'Test': [224, 308, 239, 3],
                'Val': [224, 229, 239, 3]
                },
        'Open_718_3D': {
                'Test': [301, 390, 191, 3],
                'Val': [224, 390, 191, 3]
                },
        'Ti64_Bimodal_Skipped':{
                'FullWrite': [380, 577, 534, 3]
                },
        'Ti64_Bimodal_Unskipped':{
                'Test': [281, 210, 301, 3]
                }
 
        }


sr_3d = {}
    
def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    if material == 'Ti64_3D' or material == 'Ti64_3D_LR':
        int_part = filename.split("_")[7]
    elif material == 'Open_718_3D':
        int_part = filename.split("_")[6]
    else: 
        int_part = filename.split("_")[8]

    return int(int_part) 


def npy_to_zplanes(x_dict):
    #import pdb; pdb.set_trace()
    for plane, imgs_plane in x_dict.items():
        save_path = f'{fdir}/zplanes_eval_{plane}'

        if not os.path.exists(save_path):
            os.mkdir(f'{save_path}')
             
        for img_no in range(imgs_plane.shape[0]):
             img =  imgs_plane[img_no, ...].astype(np.uint8)
             imageio.imwrite(f'{save_path}/{material}_{data}_{network}_{loss_type}_{plane}_zplane_{datatype}_{img_no}.png', img)  
 

def main():
    import pdb; pdb.set_trace()
    for plane in planes:
        sr_3d[plane] = np.zeros(sizes[material][data])
        if datatype == 'LR':
            files = sorted(glob.glob(f'{fdir}/*{plane}*_LR*.png'), key=get_key) 
        elif datatype == 'HR':
            files = sorted(glob.glob(f'{fdir}/*{plane}*_HR.png'), key=get_key) 
        else:
            files = sorted(glob.glob(f'{fdir}/*{plane}*_SR*.png'), key=get_key) 

        for i, fname in enumerate(files): 
            sr= imageio.imread(f'{fname}')
            if plane == 'X_normal' or plane == 'x_normal':
                sr_3d[plane][:,:,i,:] = sr
            elif plane == 'Y_normal' or plane == 'y_normal':
                sr_3d[plane][:,i, :,:] = sr
    
    npy_to_zplanes(sr_3d)
            

if __name__ == "__main__":
    main()

