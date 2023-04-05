#!/bin/bash

modelname="qrbsa_layernorm_rotdist_1D_x4_titanium_progpatch"
modeltoload="model_best"
filetype="HR"

#datasettype=("Val" "Test")
datasettype=("Test")

#datasettype=("FullWrite")


#materials=("Open_718_3D")

materials=("Ti64_Bimodal_Unskipped")

#materials=("Ti64_3D")
#materials=("Ti7_1Percent_3D" "Ti7_3Percent_3D")

#sect=("x_normal")
sect=("x_normal" "y_normal")

#sect=("X_normal" "Y_normal")

#sect=("combined_zplane")


for material in ${materials[@]};do

	for d_type in ${datasettype[@]}; do
		for s in ${sect[@]}; do
			echo "$d_type  $s" 

			echo "Running Numpy to Dream3D"
			python npy_to_dream3d.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
			
			echo "Changing Variable in JSON "

			python change_var_in_json.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
			

	  
			echo "Running Dream 3D Pipeline"

			#path to Dream3D program
			cd /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping/DREAM3D-6.5.141-Linux-x86_64/bin 
			
			
			./PipelineRunner -p /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping/pipeline.json

			echo "Running Dream 3D Pipeline"
			
			cd /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping 
			
            python dream3d_to_rgb.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
		


		done
	done

done
