# Holds functions for running whole brain statistical tests as well as for generating movies and images of results

import glob
import os
from pathlib import Path
import pickle
from time import time
from typing import Sequence

import numpy as np
import matplotlib.cm
import imageio
import pandas as pd

from janelia_core.dataprocessing.dataset import ROIDataset
from janelia_core.utils.data_saving import append_ts
from janelia_core.stats.regression import grouped_linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.visualization.custom_color_maps import generate_normalized_rgb_cmap
from janelia_core.visualization.volume_visualization import make_z_plane_movie

from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import extract_transitions
from keller_zlatic_vnc.linear_modeling import one_hot_from_table


def whole_brain_single_ref_testing(data_file: Path, test_type: str, cut_off_time: float, manip_type: str,
                                   save_folder: Path, save_str: str, min_n_subjects_per_beh: int = 3,
                                   beh_ref: str = 'Q', alpha: float = .05) -> Path:
    """ Runs tests of a particular type across all voxels in the brain, comparing one condition vs another.

    Test results will be saved in a file.

    Args:

        data_file: The file with the extracted dff for each voxel as well as event information, as produced
        by the folder dff_extraction.ipynb

        test_type: The type of test to run.  Should be one of the following strings:

            state_dependence - tests if dff after manipulation is sensitive to behavior before
            prediction_dependence - tests if dff before manipulation is sensitive to behavior after
            decision_dependence - tests if dff during manipulation is sensitive to behavior after
            before_reporting - tests if dff before manipulation is sensitive to behavior before
            after_reporting - tests if dff after manipulation is sensitive to behavior after

        cut_off_time: The cut off time in seconds for determining when a succeeding behavior is quiet

        manip_type: The manipulation location of events to consider.  Can either be:
            'A4' - only fits models to A4 manipulation events
            'A9' - only fits models to A9 manipulation events
            'both' - fits models to both A4 and A9 manipulation events

        save_folder: The folder where results should be stored

        save_str: A descriptive string to include in the results file name

        min_n_subjects_per_beh: In order to include a behavior in a test, it must be present in at least this many
        different subjects

        beh_ref: The behavior to compare to

        alpha: The alpha level for thresholding significance

    Returns:

        The path to the saved results.

    Raises:
        ValueError: If manip_type is not one of the expected strings.
        ValueError: If test_type is not one of the expected strings.

    """

    if (manip_type != 'A4') or (manip_type != 'A9') or (manip_type != 'both'):
        raise(ValueError('manip_type must be one of the following strings: A4, A9, both'))

    # Load data
    with open(data_file, 'rb') as f:
        file_data = pickle.load(f)
    data = file_data['event_annots']

    # Rename a few columns
    data.rename(columns={'Smp ID': 'subject_id', 'Beh Before': 'beh_before', 'Beh After': 'beh_after'}, inplace=True)

    # Apply cut off time
    _, data = extract_transitions(data, cut_off_time)

    # Down select for manipulation target if needed
    if manip_type == 'A4':
        data = data[data['Tgt Site'] == 'A4']
    elif manip_type == 'A9':
        data = data[data['Tgt Site'] == 'A9']

    # Remove behaviors which are not present in enough subjects
    trans_subj_cnts = count_unique_subjs_per_transition(data)

    if (test_type == 'state_dependence') or ('test_type' == 'before_reporting'):
        after_beh_th = 0
        before_beh_th = min_n_subjects_per_beh
    elif ((test_type == 'prediction_dependence') or (test_type == 'after_reporting') or
          (test_type == 'decision_dependence')):
        after_beh_th = min_n_subjects_per_beh
        before_beh_th = 0
    else:
        raise (ValueError('The test_type ' + test_type + ' is not recognized.'))

    after_beh_sum = trans_subj_cnts.sum()
    after_behs = [b for b in after_beh_sum[after_beh_sum >= after_beh_th].index]

    before_beh_sum = trans_subj_cnts.sum(1)
    before_behs = [b for b in before_beh_sum[before_beh_sum >= before_beh_th].index]

    before_keep_rows = data['beh_before'].apply(lambda x: x in set(before_behs))
    after_keep_rows = data['beh_after'].apply(lambda x: x in set(after_behs))
    data = data[before_keep_rows & after_keep_rows]

    # Update our list of before and after behaviors. We do this since by removing rows, some of
    # our control behaviors may no longer be present.

    new_trans_sub_cnts = count_unique_subjs_per_transition(data)
    new_after_beh_sum = new_trans_sub_cnts.sum()
    after_behs = [b for b in new_after_beh_sum[new_after_beh_sum > 0].index]
    new_before_beh_sum = new_trans_sub_cnts.sum(1)
    before_behs = [b for b in new_before_beh_sum[new_before_beh_sum > 0].index]
    print('Using the following before behaviors: ' + str(before_behs))
    print('Using the following after behaviors: ' + str(after_behs))
    print(['Number of rows remaining in data: ' + str(len(data))])

    # Pull out Delta F/F
    if (test_type == 'state_dependence') or ('test_type' == 'after_reporting'):
        dff = np.stack(data['dff_after'].to_numpy())
        print('Extracting dff after the manipulation.')
    elif (test_type == 'prediction_dependence') or (test_type == 'before_reporting'):
        dff = np.stack(data['dff_before'].to_numpy())
        print('Extracting dff before the manipulation.')
    elif test_type == 'decision_dependence':
        dff = np.stack(data['dff_during'].to_numpy())
        print('Extracting dff during the manipulation.')
    else:
        raise (ValueError('The test_type ' + test_type + ' is not recognized.'))

    # Find grouping of data by subject
    unique_ids = data['subject_id'].unique()
    g = np.zeros(len(data))
    for u_i, u_id in enumerate(unique_ids):
        g[data['subject_id'] == u_id] = u_i

    # Define a function for calculating stats
    def stats_f(x_i, y_i, g_i, alpha_i):
        beta, acm, n_grps = grouped_linear_regression_ols_estimator(x=x_i, y=y_i, g=g_i)
        stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_grps, alpha=alpha_i)
        stats['beta'] = beta
        return stats

    # Fit models and calculate stats
    before_behs_ref = list(set(before_behs).difference(ps['beh_ref']))
    after_behs_ref = list(set(after_behs).difference(ps['beh_ref']))
    before_behs_ref = sorted(before_behs_ref)
    after_behs_ref = sorted(after_behs_ref)

    n_before_behs = len(before_behs_ref)
    n_after_behs = len(after_behs_ref)

    one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(data, beh_before=before_behs_ref, beh_after=after_behs_ref)
    one_hot_data_ref = np.concatenate([one_hot_data_ref, np.ones([one_hot_data_ref.shape[0], 1])], axis=1)
    one_hot_vars_ref = one_hot_vars_ref + ['ref']

    n_rois = dff.shape[1]
    full_stats = [stats_f(x_i=one_hot_data_ref, y_i=dff[:, r_i], g_i=g, alpha_i=ps['alpha']) for r_i in range(n_rois)]

    # Package results
    if (test_type == 'state_dependence') or (test_type == 'before_reporting'):
        test_behs = before_behs_ref
        pull_inds = range(0, n_before_behs)
    elif ((test_type == 'prediction_dependence') or (test_type == 'after_reporting') or
          (test_type == 'decision_dependence')):
        test_behs = after_behs_ref
        pull_inds = range(n_before_behs, n_before_behs + n_after_behs)
    else:
        raise (ValueError('The test_type ' + test_type + ' is not recognized.'))

    beh_stats = dict()
    for b, p_i in zip(test_behs, pull_inds):
        beh_stats[b] = dict()
        beh_stats[b]['p_values'] = [rs_dict['non_zero_p'][p_i] for rs_dict in full_stats]
        beh_stats[b]['beta'] = [rs_dict['beta'][p_i] for rs_dict in full_stats]

    # Save results
    save_name = append_ts(test_type + '_' + save_str) + '.pkl'
    save_path = Path(save_folder) / save_name

    ps = {'data_file': data_file, 'test_type': test_type,  'cut_off_time': cut_off_time,
          'manip_type': manip_type, 'save_folder': save_folder, 'save_str': save_str,
          'min_n_subjects_per_beh': min_n_subjects_per_beh, 'beh_ref': beh_ref, 'alpha': alpha}

    rs = dict()
    rs['beh_stats'] = beh_stats
    rs['full_stats'] = full_stats
    rs['ps'] = ps

    with open(save_path, 'wb') as f:
        pickle.dump(rs, f)

    print('Saved results to: ' + str(save_path))

    return save_path


def make_whole_brain_videos_and_max_projs(results_file: Path, save_supp_str: str,
                                          gen_coef_movies: bool = True, gen_coef_tiffs: bool = True,
                                          gen_p_value_movies: bool = True, gen_p_value_tiffs: bool = True,
                                          gen_filtered_coef_movies: bool = True, gen_filtered_coef_tiffs: bool = True,
                                          gen_combined_movies: bool = True, gen_combined_tiffs: bool = True,
                                          gen_combined_projs: bool = True, gen_uber_movies: bool = True,
                                          p_vl_thresholds: Sequence[float] = None,
                                          coef_clim_percs: Sequence[float] = None, coef_lims: Sequence[float] = None,
                                          min_p_val_perc: float = 1.0, max_p_vl: float = .05,
                                          mean_img_clim_percs: Sequence[float] = None,
                                          ex_dataset_file: Path = None, roi_group: str = None, ):
    """ Generates movies and max projections given results of whole brain statistical tests.

    Args:
        results_file: Path to file with results of the whole-brain statistical tests.

        save_supp_str: A descriptive string to include in filenames when saving movie and image files.

        gen_coef_movies: True if movies of coefficients should be generated.

        gen_coef_tiffs: True if tiff stacks of coefficients should be generated.

        gen_p_value_movies: True if movies of p values should be generted.

        gen_p_value_tiffs: True if tiff stacks of p values should be generted.

        gen_filtered_coef_movies: True if movies of filtered coefficient values should be generated.

        gen_filtered_coef_tiffs: True if tiff stacks of filtered coefficient values should be generated.

        gen_combined_movies: True if combined movies should be generated.

        gen_combined_tiffs: True if tiff stacks of combined results should be generated.

        gen_combined_projs: True if combined projections should be generated.

        gen_uber_movies: True if movies with coefficients, p-values and tiff stacks next to eachother should be
        generated.

        p_vl_thresholds: Thresholds on p values if we are making filtered coefficient movies or tiff stacks.  If
        None, the thresholds [.05, .01, .001] will be used.

        coef_clim_percs: percentiles we use for mapping min and max coef values to colors - value should be between 0
        and 100.  This is ignored if coef_lims is used. If None, value of [1.0, 99.0] is used.

        coef_lims: specifies fixed limits for coefficients; if provided coef_clim_percs is ignored.

        min_p_val_perc: specifies lower percentile we use for mapping p-values to colors - should be between 0 and 100.
        If None, value of 1.0 is used.

        max_p_vl: specifies max p-value which is mapped to black.  If None, .05 is used.

        mean_img_clim_percs: specifies percentiles we use for mapping min and max values to colors for mean image. v
        Values should be between 0 and 100.  If None, the value [0.1, 99.9] is used.

        ex_dataset_file: Path to an example dataset to load.  We will only get the location of ROIs from this.
        Because we assume the location of ROIs is the same across all datasets, we only need to open one example
        dataset.

        roi_group: The roi group that results were generated for.

    Rasises:
        RuntimeError: If number of ROIs in results does not match the number in the specified ROI group in the inputs
        to this function

    """
    # Assign defaults to optional inputs
    if p_vl_thresholds is None:
        p_vl_thresholds = [.05, .01, .001]
    if coef_clim_percs is None:
        coef_clim_percs = [1, 99]
    if mean_img_clim_percs is None:
        mean_img_clim_percs = [0.1, 99.9]
    if ex_dataset_file is None:
        ex_dataset_file = Path(r'/groups/bishop/bishoplab/projects/keller_vnc/data/example_dataset_L2-561nm-ROIMonitoring_20170823_161308.corrected/dataset.pkl')
    if roi_group is None:
        raise(ValueError('roi_group must be assigned'))

    # Parse the file path to get the parent path and name
    results_folder = results_file.parent
    results_filename = results_file.name

    # Load the results
    rs_file = Path(results_file)
    with open(rs_file, 'rb') as f:
        rs = pickle.load(f)

    test_behs = list(rs['beh_stats'].keys())
    n_rois = len(rs['beh_stats'][test_behs[0]]['p_values'])

    # Load a dataset. Because the rois are in the same location for each dataset, we can just look at the
    # first dataset to find the location of the ROIS

    with open(ex_dataset_file, 'rb') as f:
        dataset = ROIDataset.from_dict(pickle.load(f))

    rois = dataset.roi_groups[roi_group]['rois']
    if len(rois) != n_rois:
        raise (RuntimeError('Number of rois in dataset does not match number of rois statistics are calculated for.'))

    # Load mean image
    mn_img = np.ones([2000, 1000, 20]) #dataset.stats['mean']

    # Define helper functions
    def coef_clims(vls, perc):
        if coef_lims is not None:
            return coef_lims
        else:
            small_v = np.percentile(vls, perc[0])
            large_v = np.percentile(vls, perc[1])
            v = np.max([np.abs(small_v), np.abs(large_v)])
            return [-v, v]

    def p_vl_clims(vls, perc):
        small_v = np.percentile(vls, perc)
        return [small_v, np.log10(max_p_vl)]

    def generate_norm_map():
        base_map = matplotlib.cm.viridis
        return generate_normalized_rgb_cmap(base_map, 10000)

    # Create folder to save results
    image_folder = results_file.stem  # Save images under a folder with the same name as the results
    save_folder_path = Path(results_folder) / (image_folder + save_supp_str)
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path)
    print('Saving movies and images into: ' + str(save_folder_path))

    # Save the mean image
    mn_image_path = save_folder_path / 'mean.tiff'

    imageio.mimwrite(mn_image_path, mn_img)

    mn_img_min_c_lim = np.percentile(mn_img, mean_img_clim_percs[0])
    mn_img_max_c_lim = np.percentile(mn_img, mean_img_clim_percs[1])

    make_z_plane_movie(volume=mn_img, save_path=str(save_folder_path / 'mean.mp4'),
                       cmap='gray', clim=(mn_img_min_c_lim, mn_img_max_c_lim),
                       title='Mean Image', cbar_label='$F$')