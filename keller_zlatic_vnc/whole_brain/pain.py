""" Tools for fitting statistical models of neural responses to the pain stimulus. """

import glob
import itertools
from pathlib import Path
import pickle

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from janelia_core.dataprocessing.dataset import ROIDataset
from janelia_core.stats.permutation_tests import paired_grouped_perm_test
from keller_zlatic_vnc.data_processing import calc_dff
from keller_zlatic_vnc.data_processing import generate_standard_id_for_full_annots
from keller_zlatic_vnc.data_processing import generate_standard_id_for_volume
from keller_zlatic_vnc.data_processing import read_full_annotations


def single_subject_pain_stats(analyze_subj: str, annot_folders: List[str], volume_loc_file: str, dataset_folder: str,
                              dataset_base_folder: str, f_ts_str: str, bl_ts_str: str, background: float,
                              ep: float, n_before_tm_pts: int, after_aligned: str, after_offset: int,
                              n_after_tm_pts: int, save_folder: str, save_name: str):
    """ A function for detecting rois with significant responses to the optogenetic stimulus.

    This function will:

        1) Will extract DFF in a window before and after each stimulus in each roi. (Because the user can select
        how far before and after the stimulus these windows are, any events where we cannot place both windows
        completely in the time of recording will be discarded.)

        2) It will then apply a statistical test to determine if there is a significant difference in DFF in
        the two windows

        3) It will then save the results in a manner which is ready to be passed to the visualization function
        make_whole_brain_videos_and_max_projs

    Args:

        analyze_subj: The string identifying the subject we analyze

        annot_folders: A list of folders containing the annotations for subjects.  The annotations for
        analyze_subj should be in one of these.

        volume_loc_file: The excel file listing the location of the videos of neural activity for subjects.  The
        location of data for analyze_subj should be included in this file.


        dataset_folder: Subfolder containing the dataset for each subject

        dataset_base_folder: Base folder where datasets are stored

        f_ts_str: Specify the particular ts_data key in the dataset that we should pull fluorescence from for
        calculating Delta F/F

        bl_ts_str: Specify the particular ts_data key in the dataset that we should pull baselines from for
        calculating Delta F/F

        background: The background value to use in Delta F/F calculations

        ep: The epsilon value to use in Delta F/F calculations

        n_before_tm_pts: Length of window we pull dff in from before the stimulus

        after_aligned: Specify if we align the after window to the end of the stimulus or the beginning of the
        stimulus, can be either 'start' or 'end'

        after_offset: Offset from the start of the window for dff after the event and the last stimulus timep point.
        An offset of 0, means the first time point in the window will be the last time point the stimulus was delivered

        n_after_tm_pts: Length of window we pull dff in from after the stimulus

        save_folder: Folder where we should save results

        save_name: Name of file to save results in

    """

    # ==================================================================================================================
    # Get list of all subjects we can analyze
    #  These are those we have registered volumes for and annotations

    # Get list of all annotation files and the subjects they correspond to
    annot_file_paths = list(itertools.chain(*[glob.glob(str(Path(folder) / '*.csv'))
                                              for folder in annot_folders]))
    annot_file_names = [Path(p).name for p in annot_file_paths]
    annot_subjs = [generate_standard_id_for_full_annots(fn) for fn in annot_file_names]

    # Read in location of all registered volumes
    def c_fcn(str):
        return str.replace("'", "")
    converters = {0: c_fcn, 1: c_fcn}
    volume_locs = pd.read_excel(volume_loc_file, header=1, usecols=[1, 2], converters=converters)

    volume_subjs = [generate_standard_id_for_volume(volume_locs.loc[i,'Main folder'], volume_locs.loc[i, 'Subfolder'])
                    for i in volume_locs.index]
    volume_inds = [i for i in volume_locs.index]

    # ==================================================================================================================
    # Determine where the annotation and volume data is for the subject we analyze

    volume_i = np.argwhere(np.asarray(volume_subjs) == analyze_subj)[0][0]
    annot_i = np.argwhere(np.asarray(annot_subjs) == analyze_subj)[0][0]

    volume_main_folder = volume_locs.loc[volume_inds[volume_i], 'Main folder']
    volume_sub_folder = volume_locs.loc[volume_inds[volume_i], 'Subfolder']
    annot_file = annot_file_paths[annot_i]

    # ==================================================================================================================
    # Read in the annotation data
    annotations = read_full_annotations(annot_file)

    # ==================================================================================================================
    # Down select to only stimulus events
    keep_inds = [i for i in annotations.index if annotations['beh'][i] == 'S']
    annotations = annotations.iloc[keep_inds]

    # ==================================================================================================================
    # Now we read in the $\frac{\Delta F}{F}$ data for the subject

    print('Gathering neural data for subject.')

    dataset_path = (Path(dataset_base_folder) / volume_main_folder / volume_sub_folder /
                    Path(dataset_folder) / '*.pkl')
    dataset_file = glob.glob(str(dataset_path))[0]

    with open(dataset_file, 'rb') as f:
        dataset = ROIDataset.from_dict(pickle.load(f))

    # Calculate dff
    f = dataset.ts_data[f_ts_str]['vls'][:]
    b = dataset.ts_data[bl_ts_str]['vls'][:]
    dff = calc_dff(f=f, b=b, background=background, ep=ep)

    extracted_dff = dict()
    for index in annotations.index:
        event_start = annotations['start'][index]
        event_stop = annotations['end'][index]

        dff_before = np.mean(dff[event_start - n_before_tm_pts:event_start, :], axis=0)

        if after_aligned == 'start':
            after_start_ind = event_start + after_offset
        elif after_aligned == 'end':
            after_start_ind = event_stop + after_offset
        else:
            raise ('Unable to recogonize value of after_aligned.')
        after_stop_ind = after_start_ind + n_after_tm_pts

        dff_after = np.mean(dff[after_start_ind:after_stop_ind, :], axis=0)

        extracted_dff[index] = (dff_before, dff_after)

    # ==================================================================================================================
    # Remove any events where the $\Delta F /F$ window fell outside of the recorded data

    bad_keys = [k for k, vl in extracted_dff.items() if np.all(np.isnan(vl[0]))]
    for key in bad_keys:
        del extracted_dff[key]

    # Drop same events in annotations, even though we don't use this table anymore, just for good house keeping
    annotations.drop(bad_keys, axis='index', inplace=True)

    # ==================================================================================================================
    # Calculate statistics

    dff_before = np.stack([extracted_dff[i][0] for i in extracted_dff.keys()])
    dff_after = np.stack([extracted_dff[i][1] for i in extracted_dff.keys()])

    n_rois = dff.shape[1]
    mn_stats = [None] * n_rois
    for roi_i in range(n_rois):
        mn_stats[roi_i] = _mean_t_test(dff_before[:, roi_i], dff_after[:, roi_i])
        if roi_i % 50000 == 0:
            print('Done with roi ' + str(roi_i) + ' of ' + str(n_rois) + '.')

    # ==================================================================================================================
    # Package results

    diff_vls = np.zeros(n_rois)
    p_values = np.ones(n_rois)  # Default value is 1, which is what we leave if we couldn't calculate a p-value b/c the
                                # means before and after stimulus were too close

    for roi_i in range(n_rois):
        diff_vls[roi_i] = mn_stats[roi_i]['after_mn'] - mn_stats[roi_i]['before_mn']
        if not (np.isnan(mn_stats[roi_i]['p'])):
            p_values[roi_i] = mn_stats[roi_i]['p']
        else:
            p_values[roi_i] = 1.0

    beh_stats = {'G_G': {'beta': diff_vls, 'p_values': p_values}}   # G_G = "grouped to grouped" condition transitions,
                                                                    # since we don't care what came before or after
                                                                    # stimulus
    full_stats = {'beh_stats': beh_stats}

    # ==================================================================================================================
    # Save results

    ps = {'annot_folders': annot_folders,
          'volume_loc_file': volume_loc_file,
          'analuze_subj': analyze_subj,
          'dataset_base_folder': dataset_base_folder,
          'datset_folder': dataset_folder,
          'f_ts_str': f_ts_str,
          'bl_ts_str': bl_ts_str,
          'background': background,
          'ep': ep,
          'n_before_tm_pts': n_before_tm_pts,
          'after_aligned': after_aligned,
          'after_offset': after_offset,
          'n_after_tm_pts': n_after_tm_pts,
          'save_folder': save_folder,
          'save_name': save_name}

    rs = {'ps': ps, 'full_stats': full_stats}

    save_path = Path(ps['save_folder']) / ps['save_name']
    with open(save_path, 'wb') as f:
        pickle.dump(rs, f)


# Define some helper functions here


def _mean_t_test(before_vls, after_vls):
        before_mn = np.mean(before_vls)
        after_mn = np.mean(after_vls)
        _, p = ttest_rel(a=before_vls, b=after_vls)
        return {'after_mn': after_mn, 'before_mn': before_mn, 'p': p}


def _mean_perm_test(before_vls, after_vls, n_perms):
        before_mn = np.mean(before_vls)
        after_mn = np.mean(after_vls)
        _, p = paired_grouped_perm_test(x0 = before_vls, x1 = after_vls,
                                        grp_ids=np.arange(len(after_vls)), n_perms=n_perms)
        return {'after_mn': after_mn, 'before_mn': before_mn, 'p': p}

