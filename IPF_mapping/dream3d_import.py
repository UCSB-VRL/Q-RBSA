import h5py
import numpy as np

def create_dream3d_file(source_path, output_path, attribute_container='DataContainerBundles', dream3d_version="6.2.327.4fc644a ", file_version="7.0 "):
    """Creates a new DREAM3D file with basic attributes from the file located at source_path
    and writes this file to the destination at output_path"""
    with h5py.File(output_path, 'w') as output:
        # file attributes so DREAM.3D can read the file
        output.attrs['DREAM3D Version'] = np.string_(dream3d_version)
        output.attrs['FileVersion'] = np.string_(file_version)

        # Copying of basline file attribs
        source = h5py.File(source_path)
        source.copy('Pipeline', output)

        if attribute_container in source.keys():
            source.copy(attribute_container, output)
        else:
            print('attribute_container not found')
            return 0

        return output

def create_empty_container(output_path, output_internal_path, tuple_dimensions, attribute_matrix_type=3):
    """ Creates a new dataset container at the location specified by internal_path.
    tuple_dimensions are the dimensions of the objects to be stored, excluding channel depth.
    For example, a container for a 50 X 50 array of cells would (50, 50), and within this container, additional objects
    can be stores with differing depths (e.g. Euler angles have a depth of 3 and quaternions have a depth of 4).
    attribute_matrix_type is set to be for cell-based data by default, but several int-based flag options are available
    per DREAM3D specifications.
    Possible attribute_matrix_types are:
    0 = Vertex
    1 = Edge
    2 = Face
    3 = Cell
    4 = VertexFeature
    5 = EdgeFeature
    6 = FaceFeature
    7 = CellFeature
    8 = VertexEnsemble
    9 = EdgeEnsemble
    10 = FaceEnsemble
    11 = CellEnsemble
    12 = Metadata
    13 = Generic
    999 = Unknown
    For more information on Attribute Matrix Types, consult DREAM3D documentation
    """
    with h5py.File(output_path, 'r+') as current_file:
        if isinstance(output_internal_path, str):
            new_container = current_file.create_group(output_internal_path)
            new_container.attrs['AttributeMatrixType'] = np.uint32(attribute_matrix_type)
            new_container.attrs['TupleDimensions'] = np.flip(np.uint64((list(tuple_dimensions))))
        else:
            print('No container added. internal_path must be of type: str')
            return current_file
        return current_file

def copy_container(source_path, source_internal_path, output_path, output_internal_path):
    """Copies a data container from a file located at source_path to the file being written at output_path.
    internal_path variables specify the location of the file being copied and the location to which it is being
    written."""
    with h5py.File(output_path, 'r+') as current_file:
        source = h5py.File(source_path)
        source.copy(str(source_internal_path), current_file,
                    str(output_internal_path))
        return current_file

def add_to_container(output_path, output_internal_path, target_dataset, target_dataset_name, target_dataset_type="DataArray<float> "):
    """For the file being written to output_path, at the location within the file designated by
    output_internal_path, adds the dataset given in target_dataset as a single item in the DREAM3D file.
    This dataset will be assigned the name given by target_dataset_name.  The dataset will not be added
    to the file if it is incompatible with the TupleDimensions for the assigned attribute folder, as this
    results in storage and compatibility issues within the DREAM3D file."""
    with h5py.File(output_path, 'r+') as current_file:
        # Need to verify matching dimensions between target set and attribute folder (for proper storage)
        # Possible conditions for target_dataset with proper dimensions:
        # Channel depth is 1, set is squeezed
        # Channel depth is 1 or more, set is unsqueezed
        attribute_folder = current_file[output_internal_path]
        allowed_dims = attribute_folder.attrs.get('TupleDimensions')
        num_tuple_dims = len(allowed_dims)
        target_dims = np.asarray(list(np.shape(target_dataset)))
        num_target_dims = len(target_dims)
        if num_tuple_dims==num_target_dims and np.array_equal(np.flip(allowed_dims),target_dims):
            # Channel depth = 1, squeezed
            added_set = current_file.create_dataset(str(output_internal_path)+'/'+str(target_dataset_name),
                                                    data=target_dataset)
            added_set.attrs['ComponentDimensions'] = np.uint64([1])
        elif num_tuple_dims==num_target_dims-1 and np.array_equal(np.flip(allowed_dims), target_dims[0:num_tuple_dims]):
            added_set = current_file.create_dataset(str(output_internal_path) + str('/') + str(target_dataset_name),
                                                    data=target_dataset)
            added_set.attrs['ComponentDimensions'] = np.uint64([target_dims[-1]])
        else:
            print('target_dataset dimensions must be compatible with container dimensions')
            return current_file
        if num_tuple_dims==1:
            added_set.attrs['Tuple Axis Dimensions'] = np.string_(
                "x=" + str(allowed_dims[0])+" ")
        if num_tuple_dims==2:
            added_set.attrs['Tuple Axis Dimensions'] = np.string_(
                "x=" + str(allowed_dims[0]) + ",y=" + str(allowed_dims[1])+" ")
        if num_tuple_dims==3:
            added_set.attrs['Tuple Axis Dimensions'] = np.string_(
                "x=" + str(allowed_dims[0]) + ",y=" + str(allowed_dims[1]) + ",z=" + str(allowed_dims[2]) + " ")
        added_set.attrs['DataArrayVersion'] = np.int32([2])
        added_set.attrs['ObjectType'] = np.string_(target_dataset_type)
        added_set.attrs['TupleDimensions'] = np.uint64([allowed_dims])
        return current_file

def create_geometry_container_from_source(source_path, output_path, dimensions=None, origin=None, spacing=None,
                                    source_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY',
                                    output_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY'):
    """This is a function specifically for the building of the geometry dataset container, _SIMPL_GEOMETRY which has a
    custom layout that is consistent across all DREAM3D data files.  For this reason, the only required inputs are the
    source file location (source_path) and the DREAM3D file being written (dream3d_file).  If no dimensions, origin, or
    spacing are specified, the dimensions, origin, and spacing will be copied form the source file.  Internal paths to
    the _SIMPL_GEOMETRY folder are set to the default DREAM3D file locations, so if any internal folders are renamed in
    the source file or the file being written, the paths to the _SIMPL_GEOMETRY folder must be specified manually.
    *******************************************
    NOTE CAREFULLY! The DREAM3D folder _SIMPL_GEOMETRY does NOT contain the full spelling of the word simple, as there
    is a missing letter E!  This typo can lead to file path errors and DREAM3D files that cannot be properly read.
    Please check your spelling of the internal path designations for this folder.
    *******************************************
    """
    with h5py.File(output_path, 'r+') as current_file:
        source = h5py.File(source_path)
        if isinstance(source_internal_geometry_path, str) and isinstance(output_internal_geometry_path, str):
            source.copy(source_internal_geometry_path, current_file,
                    output_internal_geometry_path)
            source_dimensions_path = str(source_internal_geometry_path)+str('/DIMENSIONS')
            output_dimensions_path = str(output_internal_geometry_path)+str('/DIMENSIONS')
            if dimensions is not None:
                current_dimensions = current_file[output_dimensions_path]
                dimensions = list(dimensions)
                current_dimensions[...] = np.flip(np.uint64(dimensions))

            source_origin_path = str(source_internal_geometry_path)+str('/ORIGIN')
            output_origin_path = str(output_internal_geometry_path)+str('/ORIGIN')
            if origin is not None:
                current_origin = current_file[output_origin_path]
                origin = list(origin)
                current_origin[...] = np.flip(np.float32(origin))

            source_spacing_path = str(source_internal_geometry_path)+str('/SPACING')
            output_spacing_path = str(output_internal_geometry_path)+str('/SPACING')
            if spacing is not None:
                current_spacing = current_file[output_spacing_path]
                spacing = list(spacing)
                current_spacing[...] = np.flip(np.float32(spacing))

        return current_file
