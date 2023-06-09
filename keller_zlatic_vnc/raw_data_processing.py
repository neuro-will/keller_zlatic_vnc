""" Code to go from videos to extracted dff. """

import os
import multiprocessing as mp
from typing import List
import pathlib
import pickle
import time

import h5py
import numpy as np
import pyspark

from janelia_core.cell_extraction.roi import extract_rois
from janelia_core.cell_extraction.super_voxels import extract_super_voxels_in_brain
from janelia_core.dataprocessing.baseline import percentile_filter_multi_d
from janelia_core.dataprocessing.roi import ROI
from janelia_core.fileio.exp_reader import find_images


def generate_rois_from_segments(seg_image: np.ndarray) -> List[ROI]:
    """ Generates rois from segments.

    Args:

        seg_image: A 3-d volume with labeled segments.  0 indicates background. Integer values correspond to segments.

    Returns:
        rois: A list of roi objects representing all segments in the object.
    """

    # The id denoting background
    BG_ID = 0

    # Find unique segment ids, removing the background ID
    seg_ids = np.unique(seg_image)
    seg_ids = seg_ids[seg_ids != BG_ID]

    # Generate rois
    n_cpu = mp.cpu_count()
    with mp.Pool(n_cpu) as pool:
        rois = pool.starmap(_form_roi, [(seg_image, id) for id in seg_ids])

    return rois


def video_to_roi_baselines(base_data_dir: pathlib.Path, save_dir: pathlib.Path, roi_extract_opts: dict,
                                  baseline_calc_opts: dict, extract_params: dict,
                                  img_file_ext: str = 'weightFused.TimeRegistration.templateSpace.klb',
                                  new_comp: bool = False, sc:pyspark.SparkContext = None,
                                  roi_vl_file_name: str = 'extracted_f.h5',
                                  roi_desc_file_name: str = 'roi_locs.pkl',
                                  baseline_file_name: str = 'baseline_f.h5',
                                  extract_params_file_name: str = 'extraction_params.pkl',
                                  rois: List[ROI] = None):
    """ Pipeline to go from videos to extraced F and baseline F for ROIS in Keller/Zlatic vnc data.

    This function will:

        1) Extract average (across space) intensity in ROIS across space.  It can do this in one of two ways.

            1) The default is to place supervoxels across space of a specified size in a given brain mask and
            extract fluorescence and baselines values for each supervoxel.

            2) Alternatively, the user can specify a set of rois.  These ROIs can each have a custom shape and
            placement, and then fluorescence and baselines are only calculated for these ROISs.

        2) Calculate a baseline for each ROI

        4) Save results

    This function is designed so it can "pick up where it left off."  This means intermediate
    results will be saved for step 1 above, and if it is found that ROIS have already been
    extracted, by default the function will use those existing results and move directly to
    calculate baselines. The user can change this behavior so that results are performed fresh
    no matter what.

    Args:
        base_data_dir: The base directory for the dataset.  This is the directory containing
        folders for each time point in the dataset under which image data is stored.

        save_dir: The directory that results should be saved in.

        roi_extract_opts: A dictionary of options to pass to either
            1) extract_super_voxels_in_brain (if extracting super voxels).  At a minimum, it should contain the key
            'voxel_size_per_dim'.

            2) extract_rois (if extracting prespecified ROIs).  There are no required options in this case.

        baseline_calc_opts: A dictionary of options to pass to percentile_filter_multi_d to calculate baselines.
        Must include window_length, filter_start, write_offset, p, and n_processes.

        extract_params: A dictionary with parameters that were used for extraction - these will be saved with the
        data to have a record of the settings that were used

        image_ext: The extension for the type of image files to load.

        new_comp: If False, any intermediate results that are found will be used.  If true,
        computations will be performed from scratch, and existing intermediate results will
        be overwritten.

        sc: An optional spark context to use to speed up computation.

        roi_vl_file_name: The name of the hdf5 file where the extracted intensity of rois
        will be saved.

        roi_desc_file_name: The name of the pickle file where the description information for the rois
                            will be saved.

        baseline_file_name: The name of the hdf5 file where the calculated baseline of rois will be saved

        extract_params_file_name: The name of the pickle file that extraction parameters should be saved in.

            Note: If extraction parameters include a preprocessing function, this will be removed before saving
            the pickled parameters.

    """

    # First, we create the save directory if we need to
    if os.path.exists(save_dir):
        print('Save directory already exists: ' + str(save_dir))
    else:
        print('Save directory does not already exist.  Creating: ' + str(save_dir))
        os.makedirs(save_dir)

    # Now we extract supervoxels
    print('==================================================================')
    print('Beginning supervoxel extraction.')
    print('==================================================================')

    # See if supervoxels have already been extracted
    roi_vl_file = save_dir / roi_vl_file_name
    roi_desc_file = save_dir / roi_desc_file_name

    skip_roi_extraction = os.path.exists(roi_vl_file) and os.path.exists(roi_desc_file) and not new_comp

    if skip_roi_extraction:
        print('ROIs have already been extracted.  Using existing ROI information saved in: ')
        print(str(roi_vl_file))
        print(str(roi_desc_file))

        with h5py.File(roi_vl_file, 'r') as f:
            roi_vls = f['data'][:]
    else:

        # Find the images for this dataset
        imgs = find_images(image_folder=base_data_dir, image_ext=img_file_ext, image_folder_depth=1)

        # Extract ROIs
        extract_t0 = time.time()
        if rois is None:
            # If no ROIS provided, we use supervoxels
            roi_vls, rois = extract_super_voxels_in_brain(images=imgs, sc=sc, **roi_extract_opts)
        else:
            roi_vls = extract_rois(images=imgs, rois=rois, sc=sc, **roi_extract_opts)

        extract_t1 = time.time()
        n_extracted_rois = len(rois)
        print('Extracted ' + str(n_extracted_rois) + ' ROIS in ' + str(extract_t1 - extract_t0) + ' seconds.')

        # Save extracted roi information
        with h5py.File(roi_vl_file, 'w') as f:
            f.create_dataset('data', data=roi_vls)

        roi_dicts = [r.to_dict() for r in rois]
        with open(roi_desc_file, 'wb') as f:
            pickle.dump(roi_dicts, f)

    # Now we calculate baselines
    print('==================================================================')
    print('Beginning baseline calculation.')
    print('==================================================================')

    baseline_file = save_dir / baseline_file_name
    skip_baseline_calcs = os.path.exists(baseline_file) and not new_comp
    if skip_baseline_calcs:
        print('Baselines have already been calculated.  Using baselines saved in: ')
        print(str(baseline_file))
    else:
        baseline_t0 = time.time()
        baseline_vls = percentile_filter_multi_d(roi_vls, **baseline_calc_opts)
        baseline_vls = baseline_vls.astype('float32')
        baseline_t1 = time.time()
        print('Baselines calculated in ' + str(baseline_t1 - baseline_t0) + ' seconds.')

        # Save extracted baseline information
        with h5py.File(baseline_file, 'w') as f:
            f.create_dataset('data', data=baseline_vls)

    # Now we save extraction parameters
    param_save_file = save_dir / extract_params_file_name

    if 'preprocess_f' in extract_params.keys():
        extract_params['preprocess_f'] = 'not recorded'

    with open(param_save_file, 'wb') as f:
        pickle.dump(extract_params, f)


# Helper functions


def _form_roi(seg_image, seg_id):
    vls = np.argwhere(seg_image == seg_id)
    roi = ROI(voxel_inds=tuple(vls[:, c] for c in range(3)), weights=np.ones(vls.shape[0]))
    roi.seg_id = seg_id
    return roi





