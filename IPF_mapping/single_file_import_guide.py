import h5py
import numpy as np
import dream3d_import as d3
import glob
import os
from argparser import Argparser

import pdb; pdb.set_trace()

args = Argparser().args


npy_file_dir = f'../experiment/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))

total_file = len(file_locs)

arr_list = []
for file_loc in file_locs:
    arr = np.load(file_loc)
    arr_list.append(arr)

import pdb; pdb.set_trace()
loaded_npy = np.asarray(arr_list)
     



# The path for the source (ground truth Dream3D file) that your newly written file will be based on
# A good choice for this is the Ti64 DIC Homo and Cubochoric Set off the google drive, but any well-defined
# Ti64 set should work just as well.  Though to be safe, the source should be a real experimental data file.
d3_sourceName = '/data/dkjangid/superresolution/Material_Dataset/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d'

# The path for the output Dream3D file being written.  This is where you want to save the file you are making.
d3_outputName = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'

# If your initial numpy file is a 2D File, you should expand to 3D.  Dream3D technically works on 2D files, but the
# safety checks are not very robust, so it is better to be safe than sorry.  Also be careful of any processing that
# squeezes your set, as it will make Dream3D very unhappy.

#loaded_npy = np.expand_dims(np.load(npy_file_path), 2)
#loaded_npy = np.expand_dims(loaded_npy, 2)



# Load the Dream3D source data (we will need to copy metadata from this)
# This may not be strictly necessary, but h5py docs are not the clearest, and trying to operate on closed hdf5 files
# using h5py can have strange and unintended consequences.
d3source = h5py.File(d3_sourceName, 'r')

# Establish the dimensions of your loaded numpy file, these will be needed for the creation of Dream3D containers.
# Each Dream3d container has a set size based on the x, y, and z, dimensions of the objects being stored in it, with
# channel depth being treated as a tuple value that can vary.  For example, a 40 x 50 x 60 array of depth-3 Euler angles
# can be stored with an analagous array of depth-4 quaternions, so long as they both have dimensions of 40 x 50 x 60.
# All objects within a container must have the same x, y and z dimensions if they are to be stored together.
# If you try and write something of a different size, you will either get empty padding (too small),
# or truncation (too big).
xdim,ydim,zdim,channeldepth = np.shape(loaded_npy)

# Creation of a Phases variable.  This is a Dream3D specific flag variable that specifies which phases of material
# are where within your dataset.  It is always an int32 variable where 0 is treated as void, and every number after
# zero is a phase whose descriptions are given in the metadata for that Dream3D file.  All of our materials (for now)
# are single phase, so this is just a big box of ones that is the size of your dataset.
phases = np.int32(np.ones((xdim,ydim,zdim)))

# Creation of the Dream3D file.  This function specifically establishes some core file properties and builds the
# DataContainers folder, which is basically the parent directory for every Dream3D file. Every folder within
# DataContainers represents a different "dataset" that goes with the associated project.  So for example, if I had
# two different sized numpy files that I wanted in the same Dream3D file, they would get different folders within
# the DataContainers directory.
new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)


# Creation of a set folder.  Default set name: ImageDataContainer
# The next few steps are a bit confusing, mostly because Dream3D is very efficient in it's storage structure.
# Even though datasets are defined by the folders in DataContainers, these folders themselves are only really defined
# in the hdf5 file by the properties within them, so no function is really needed to make an empty folder, you can
# just create a directory label and start writing to it.  Being honest, I am not sure how this data is stored
# (h5py stores the file directory as a dictionary, but even then, data structure isn't quite the same), but
# it ultimately gives some structure to the hdf5 array that is useful in navigating the hdf5 array, so we do it.

# Long story short, you can create any set folder you want inside DataContainers by just writing stuff to that
# directory, so in our case, we start by copying the CellEnsembleData folder in, which is basically material metadata.
new_file = d3.copy_container(d3_sourceName, 'DataContainers/ImageDataContainer/CellEnsembleData', d3_outputName,
                          'DataContainers/ImageDataContainer/CellEnsembleData')

# If you wanted to create multiple set folders, you could do so by just repeating this step with a new directory name
# For example:
# new_file = d3.copy_container(sourceName, 'DataContainers/ImageDataContainer/CellEnsembleData', outputName,
        #                              'DataContainers/MY_SET_NAME/CellEnsembleData')

# Also BE CAREFUL:  In this function, the first path is the location of CellEnsembleData in the source, so if you are
# copying from a set whose name has been changed from the default of ImageDataContainer, your call needs to change.
# The general form is:
# new_file = d3.copy_container(sourceName, 'DataContainers/SOURCE_SET_NAME/CellEnsembleData', outputName,
        #                              'DataContainers/MY_SET_NAME/CellEnsembleData')

# Creation of the _SIMPL_GEOMETRY folder.  This is the folder that Dream3D uses to orient itself in the different
# "sections" of the HDF5 file.  This folder establishes the dimensions of the set as well as its origin and the
# physical size of the voxels (for conversion from voxel to real space).  Every set folder needs its own
# _SIMPL_GEOMETRY folder, or else Dream3D will not be able to read that set.
#
# WATCH OUT FOR EASY SPELLING MISTAKES. THERE IS NO E IN THE FIRST PART OF _SIMPL_GEOMETRY!!!!
# I added that here and in the other file, because I have wasted more time than I care to admit on that typo.
#
new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim),
                            source_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY',
                            output_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY')

# Writing your array data into CellData.  Here we do need to explicitly create the CellData folder since it needs to
# have a specific size that matches the dimensions given in the _SIMPL_GEOMETRY folder.  Creation step is done with
# create_empty_container, and the writes for Quats and Phases are done using the add_to_container.  Again, phases is
# just a bunch of ones, but it's needed for IPF conversion in Dream3D.
new_file = d3.create_empty_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', (xdim,ydim,zdim), 3)
new_file = d3.add_to_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', loaded_npy, 'Quats')
new_file = d3.add_to_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', phases, 'Phases')

# Close out source file to avoid weird memory errors.
d3source.close()
