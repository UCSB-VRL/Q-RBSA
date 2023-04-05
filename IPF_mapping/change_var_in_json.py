import json
from argparser import Argparser

args = Argparser().args

f = open('pipeline.json')

field_dict = json.load(f)

data = args.data

if data == 'Ti64':
    if args.section == "X_Block":
        Tupel = [471, 142, 18]
    elif args.section == "Y_Block":
        Tupel = [471, 346, 7]
    elif args.section == "Z_Block":
        Tupel = [142, 346, 24]

elif data == 'Ti64_3D':
    # y, x, z
    Tupel = [142, 270, 120]

    #if args.section == "X_normal": 
    #    Tupel = [120, 142, 270]
    #    #Tupel = [142, 270, 120]
    #elif args.section == "Y_normal":
    #    Tupel = [120, 270, 142]
    #    #Tupel= [142, 270, 120] 
    #elif args.section == "combined_zplane":
    #    Tupel = [142, 270, 120]
       
elif data == 'Ti7_1Percent':
    if args.section == "X_Block":
        Tupel = [540, 616, 18]
    elif args.section == "Y_Block":
        Tupel = [209, 616, 46]
    elif args.section == "Z_Block":
        Tupel = [209, 540, 52]

elif data == 'Ti7_1Percent_3D':
    # y, x, z
    Tupel = [262, 202, 232]
    #if args.section == "X_normal" or "combined_zplane":
    #    Tupel = [232, 262, 202]
    #    #Tupel = [262, 202, 232]
    #elif args.section == "Y_normal":
    #    Tupel = [232, 202, 262]
    #    #Tupel = [262, 202, 232]

     
elif data == 'Ti7_3Percent':
    if args.section == "X_Block": 
        Tupel = [540, 616, 18]
    elif args.section == "Y_Block":
        Tupel = [209, 616, 46]
    elif args.section == "Z_Block":
        Tupel = [209, 540, 52]

elif data == 'Ti7_3Percent_3D':
    Tupel = [308, 239, 224]
    #if args.section == "X_normal" or "combined_zplane":
    #    Tupel = [224, 308, 239]
    #    #Tupel = [308, 239, 224]
    #elif args.section == "Y_normal":
    #    Tupel = [224, 229, 308]
    #    #Tupel = [308, 239, 224]

     
elif data == 'Ti64_Bimodal_Skipped':
        Tupel = [577, 534, 760]   

elif data == 'Ti64_Bimodal_Unskipped': 
    Tupel = [210, 301, 281] 
 
elif data =='Open_718':
    if args.section == 'x_block' :
        Tupel = [301, 390, 191]
    if args.section == 'y_block' :
        Tupel = [301, 191, 390]
    if args.section == 'z_block' :
        Tupel = [390, 191, 301]


elif data =='Open_718_3D':
    if args.section == 'x_normal' :
        Tupel = [301, 390, 191]
    if args.section == 'y_normal' :
        Tupel = [301, 191, 390]
    





'''
elif data == 'Ti64_3D':
    if args.section == "X_Block":
        Tupel = [135, 447, 28]
    elif args.section == "Y_Block":
        Tupel = [209, 616, 46]
    elif args.section == "Z_Block":
        Tupel = [209, 540, 52]


elif data == 'Ti7_1Percent':
    if args.section == "X_Block":
        Tupel = [434, 551, 9]
    elif args.section == "Y_Block":
        Tupel = [220, 551, 37]
    elif args.section == "Z_Block":
        Tupel = [220, 434, 47]

elif data == 'Ti7_3Percent':
    if args.section == "X_Block":
        Tupel = [510, 652, 18]
    elif args.section == "Y_Block":
        Tupel = [213, 652, 43]
    elif args.section == "Z_Block":
        Tupel = [213, 510, 56]

'''

#import pdb; pdb.set_trace() 
print(f'Tupel: {Tupel}')
npy_file_dir = f'/data/dkjangid/Material_Projects/superresolution/Quaternion_experiments/saved_weights/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

filedir = f'{npy_file_dir}/{args.data}/Dream3D/{args.section}_{args.file_type}.dream3d'


field_dict["0"]["InputFile"] = f'{filedir}'

field_dict["3"]["OutputFile"] = f'{filedir}'


field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][0]["Tuple Dimensions"]= Tupel

field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][1]["Tuple Dimensions"]= Tupel


outfile = open("pipeline.json", "w")

json.dump(field_dict, outfile, indent=4)

outfile.close()
