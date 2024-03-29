""" Tools for fitting statistical models to spontaneous data. """


import glob
import itertools
import os
from pathlib import Path
import pickle
from typing import Tuple

import multiprocessing as mp
import numpy as np
import pandas as pd

from janelia_core.dataprocessing.dataset import ROIDataset
from janelia_core.stats.multiple_comparisons import apply_by
from janelia_core.stats.multiple_comparisons import apply_bonferroni
from janelia_core.stats.regression import linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import grouped_linear_regression_ols_estimator

from keller_zlatic_vnc.data_processing import apply_cutoff_times
from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import count_transitions
from keller_zlatic_vnc.data_processing import calc_dff
from keller_zlatic_vnc.data_processing import find_quiet_periods
from keller_zlatic_vnc.data_processing import generate_standard_id_for_full_annots
from keller_zlatic_vnc.data_processing import generate_standard_id_for_volume
from keller_zlatic_vnc.data_processing import get_basic_clean_annotations_from_full
from keller_zlatic_vnc.data_processing import read_full_annotations
from keller_zlatic_vnc.linear_modeling import one_hot_from_table
from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import test_for_diff_than_mean_vls


def apply_multiple_comparisons_corrections(p_vls: np.ndarray, computed_p_vls: np.ndarray):
    """ Applies multiple comparisons to whole-brain statistics.

    Multiple comparison correction is applied independently across maps for each behavior.

    Args:
        p_vls: A matrix of p values to correct.  Columns are behaviors; rows are rois.

        computed_p_vls: A boolean matrix indicating which entries in p_vls we should run multiple comparisons over.
        There reason for this is that there are some behaviors and rois we may not actually fit models to.  In that
        case, we should not include those when doing multiple comparisons.

    Returns:

        corrected_p_vls_by: Benjamini-Yekutieli corrected p values.  If we left a p-value out of multiple comparisons
        correction (because it's value in computed_p_vls was False), then it's entry here will be nan.

        corrected_p_vls_bon: Bonferroni corrected p values.  Same format as corrected_p_vls_by.

    """
    corrected_p_vls_by = np.zeros(p_vls.shape)
    corrected_p_vls_by[:] = np.nan
    corrected_p_vls_bon = np.zeros(p_vls.shape)
    corrected_p_vls_bon[:] = np.nan
    for c_i in range(p_vls.shape[1]):
        cur_col_vls = p_vls[computed_p_vls[:, c_i], c_i]
        _, cur_col_corrected_p_vls_by = apply_by(p_vls=cur_col_vls,
                                                 alpha=.05)  # alpha is not used for the second output
        corrected_p_vls_by[computed_p_vls[:, c_i], c_i] = cur_col_corrected_p_vls_by

        _, cur_col_corrected_p_vls_bon = apply_bonferroni(p_vls=cur_col_vls, alpha=.05) # Alpha not used here either
        corrected_p_vls_bon[computed_p_vls[:, c_i], c_i] = cur_col_corrected_p_vls_bon

    return corrected_p_vls_by, corrected_p_vls_bon


def calc_mean_dff(x: np.ndarray, start: int, stop: int, window_type: str,
                  window_offset: int = None, window_length: int = None) -> Tuple[np.ndarray, bool, bool]:
    """Calculates mean DFF in a window.

    Args:
        x: DFF of shape n_time_pts * n_rois
        start: start index of the event
        stop: stop index of the event
        window_type: Type of window.  Either 'whole_event' or 'start_locked'.
        window_offset: The offset to the start of the window from event start if window type is start_locked.
        window_length: The length of the window if the window tpe is start_locked.

    Returns:
         mn_vls: The mean values across all rois.
         starts_within_event: True if the window starts within the event.
         stops_within_event: True if the window stops within the event.
    """

    if window_type == 'whole_event':
        take_slice = slice(start, stop)
        starts_within_event = True
        stops_within_event = True
    elif window_type == 'start_locked':
        start_offset = start + window_offset
        stop_offset = start_offset + window_length
        take_slice = slice(start_offset, stop_offset)
        starts_within_event = (stop >= start_offset) and (start <= stop_offset)
        stops_within_event = (stop >= stop_offset) and (start <= stop_offset)
    else:
        raise(ValueError('The window_type is not recogonized.'))

    if (take_slice.start < 0) or (take_slice.stop > x.shape[0]):
        mn_vls = np.nan
    else:
        mn_vls = np.mean(x[take_slice, :], axis=0)

    return mn_vls, starts_within_event, stops_within_event


def fit_init_models(ps: dict, return_dff: bool = False) -> Tuple[dict, dict, dict]:
    """ Fits initial models to spontaneous activity.

    This function will:

        1) Read in annotation and neural data for multiple subjects.  The user supplies paths to where annotation
        data as well as volume (neural) data can be provided for subjects.  Be default, all subjects for which there is
        annotation data as well as volume data will be analyzed.  (See the exclude_subjs argument for how to modify
        this).

        2) Find quiet periods between marked events for each subject.  See the function find_quiet_periods for more
        detail.

        3) Find "clean" spontaneous events for each subject.  See the function get_basic_clean_annotations_from_full
        for the options for defining clean events.

        4) Apply a cut off time threshold to classify the preceding behavior for each event.  See
        the function apply_cutoff_times for more information.

        5) Filter events by the behavior that is transitioned into and from (if requested).  Importantly, this is
        done before any pooling of behaviors is done (see 4-6).

        6) Optionally pool preceding and succeeding left and right turns into single turn behaviors.

        7) Calculate Delta F/F for all events in a specified window for each event for all supervoxels (see options
        below for specifying the placement of the window)

        8) Remove any events where the window for Delta F/F fell outside the range of recorded data

        9) Optionally, remove any events where the window for Delta F/F fell outside the start and stop of the behavior
        transitioned into

        10) Remove any transitions from consideration if we don't see enough of them in enough subjects or enough of them
        across all subjects (two seperate criteria are applied, see below)

        11) Fit statistical models on a per-voxel basis.  These statistical models estimate the dff for each transition,
        accounting for correlated errors within subjects (if fitting to more than one subject).

        12) Package the results

    Note: One subject (CW_17-11-03-L6-2) has a different name for its saved volume.  If this subject is included
    in the analysis, this function will make sure the correct volume is used.


    Args:

        ps: A dictionary with the following keys:

            annot_folders: list of folders containing a4 and a9 annotation data

            volume_loc_file: file containing locations to registered volumes

            exlude_subjs: list subjects we do not want to include in the analysis

            dataset_folder: subfolder containing the dataset for each subject

            datast_base_folder: base folder where datasets are stored

            clean_event_def: The defintion we use for clean events.  Options are:

                dj: "Clean" events are those which have *no* overlap with any other event

                po: A "clean" events can only intersect at most one other event that starts before it starts and
                    ends before it ends.

            q_th: the threshold we use (in number of stacks) to determine when a quiet transition has occurred

            co_th: The threshold we use (in number of stacks) to determine when a real transition has occured

            behs: If not None, the list of beheviors that are transitioned into to include in the analysis.  Any
            events with a behavior that is transitioned into not in this list will be removed from the analysis.

            pre_behs: If not None, the list of preceding behaviors that we transition from to include in the analysis.
            Any events with preceding behaviors not in this list will be removed from the analysis.

            pool_preceeding_behaviors: True if we want to pool preceeding behaviors

            pool_preceeding_turns: true if we want to pool preceeding left and right turns into one category (only applies if pool_preceeding_behaviors is false)

            pool_succeeding_turns: true if we want to pool succeeding left and right turns into one category

            remove_st: True if events with transitions from and to the same behavior should be removed.

            f_ts_str: a string indicating the field of timestamp data in the datasets where we will find the fluorescene
            data we are to process when forming Delta F/F

            bl_ts_str: a string indicating the field of timestamp data in the datasets where we will find the baseline
            data we are to process

            background: the background value to use when calculating dff

            ep: the epsilon value to yse when calculating dff

            min_n_subjs: min number of subjects we must observe a transition in for us to analyze it

            min_n_events: min number of times we must obesrve a transition (irrespective of which subjects we observe it
            in) for us to analyze it

            alpha: Alpha value for thresholding p-values when calculating initial stats

            window_type: The type of window we pull dff from.  Optiosn are:

                    start_locked: A fixed length window with a start aligned to the start of behavior (see window_offset
                    and window_length below for options of adjusting the position of window relative to behavior start)

                    whole_event: We analyze average dff between the start and stop of a marked behavior

            window_offset: If we are using a window locked to event start, this is the relative offset of the window
            start, relative to behavior start, in time steps

            window_length: If we are using a window locked to event start, this is the length of the window in time steps

            enforce_contained_events: If true, only analyze behaviors with start and stop times contained within the
            dff window

            save_folder: Folder we save events into. If None, results will not be saved.

            save_name: Name of the file to save results in

        return_dff: If true a dictionary with full dff time series for all volumes will be returned.  Keys correspond
            to subject ids.  Note this is very memory intensive.

    Returns:

            rs: The fitting results

            ananlyze_annotations: The annotations for all events that were used in model fitting

            dff: If return_dff is true, a dictionary with keys which are subject ids and values which are calculated
            dff traces for full volumes.  If return_dff is false, this will be an empyt dictionary.

    """

    # ==================================================================================================================
    # Get list of all subjects we can analyze

    # Get list of all annotation files and the subjects they correspond to
    annot_file_paths = list(itertools.chain(*[glob.glob(str(Path(folder) / '*.csv')) for folder in ps['annot_folders']]))
    annot_file_names = [Path(p).name for p in annot_file_paths]
    annot_subjs = [generate_standard_id_for_full_annots(fn) for fn in annot_file_names]

    # Read in location of all registered volumes
    def c_fcn(str):
        return str.replace("'", "")
    converters = {0: c_fcn, 1: c_fcn}

    volume_locs = pd.read_excel(ps['volume_loc_file'], header=1, usecols=[1, 2], converters=converters)
    volume_subjs = [generate_standard_id_for_volume(volume_locs.loc[i, 'Main folder'],
                                                    volume_locs.loc[i, 'Subfolder']) for i in volume_locs.index]
    volume_inds = [i for i in volume_locs.index]

    # Update name of one of the volume subjects to match the annotations (this is only needed for one subject)
    m_ind = np.argwhere(np.asarray(volume_subjs) == 'CW_17-11-03-L6')
    if len(m_ind) > 0:
        m_ind = m_ind[0][0]
        volume_subjs[m_ind] = 'CW_17-11-03-L6-2'

    # Produce final list of annotation subjects by intersecting the subjects we have annotations for with those we
    # have volumes before and removing any exclude subjects.
    analyze_subjs = set(volume_subjs).intersection(set(annot_subjs))
    analyze_subjs = analyze_subjs - set(ps['exclude_subjs'])
    analyze_subjs = list(np.sort(np.asarray(list(analyze_subjs))))

    # ==================================================================================================================
    # For each subject we analyze, determine where it's annotation and volume data is
    subject_dict = dict()
    for s_id in analyze_subjs:
        volume_i = np.argwhere(np.asarray(volume_subjs) == s_id)[0][0]
        annot_i = np.argwhere(np.asarray(annot_subjs) == s_id)[0][0]
        subject_dict[s_id] = {'volume_main_folder': volume_locs.loc[volume_inds[volume_i], 'Main folder'],
                              'volume_sub_folder': volume_locs.loc[volume_inds[volume_i], 'Subfolder'],
                              'annot_file': annot_file_paths[annot_i]}

    # ==================================================================================================================
    # Read in the annotation data for all subjects we analyze.  We also generate cleaned and supplemented annotations
    # here

    annotations = []
    for s_id, d in subject_dict.items():
        tbl = read_full_annotations(d['annot_file'])
        quiet_tbl = find_quiet_periods(annots=tbl, q_th=ps['q_th'], q_start_offset=ps['q_start_offset'],
                                       q_end_offset=ps['q_end_offset'])
        tbl = pd.concat([tbl, quiet_tbl], ignore_index=True)
        tbl['subject_id'] = s_id
        annotations.append(tbl)

    annotations = [get_basic_clean_annotations_from_full(annot, clean_def=ps['clean_event_def'])
                   for annot in annotations]

    annotations = pd.concat(annotations, ignore_index=True)

    # Apply the cut off time threshold
    annotations = apply_cutoff_times(annots=annotations, co_th=ps['co_th'])

    # ==================================================================================================================
    # Filter events by the behavior transitioned into or from if we are suppose to
    if ps['acc_behs'] is not None:
        keep_inds = [i for i in annotations.index if annotations['beh'][i] in ps['acc_behs']]
        annotations = annotations.loc[keep_inds]

    if ps['acc_pre_behs'] is not None:
        keep_inds = [i for i in annotations.index if annotations['beh_before'][i] in ps['acc_pre_behs']]
        annotations = annotations.loc[keep_inds]

    # ==================================================================================================================
    # Pool preceeding turns if requested
    if ps['pool_preceeding_turns']:
        turn_rows = (annotations['beh_before'] == 'TL') | (annotations['beh_before'] == 'TR')
        annotations.loc[turn_rows, 'beh_before'] = 'TC'

    # ==================================================================================================================
    # Pull succeeding turns if requested
    if ps['pool_succeeding_turns']:
        turn_rows = (annotations['beh'] == 'TL') | (annotations['beh'] == 'TR')
        annotations.loc[turn_rows, 'beh'] = 'TC'

    # ==================================================================================================================
    # Remove self transitions if requested
    if ps['remove_st']:
        self_trans = annotations['beh_before'] == annotations['beh']
        annotations = annotations.loc[~self_trans]

    # ==================================================================================================================
    # Now we read in the Delta F\F data for all subjects
    dff_traces = dict()

    extracted_dff = dict()
    for s_id in analyze_subjs:
        print('Gathering neural data for subject ' + s_id)

        # Load the dataset for this subject
        data_main_folder = subject_dict[s_id]['volume_main_folder']
        data_sub_folder = subject_dict[s_id]['volume_sub_folder']

        dataset_path = (Path(ps['dataset_base_folder']) / data_main_folder / data_sub_folder /
                        Path(ps['dataset_folder']) / '*.pkl')
        dataset_file = glob.glob(str(dataset_path))[0]

        with open(dataset_file, 'rb') as f:
            dataset = ROIDataset.from_dict(pickle.load(f))

        # Calculate dff
        f = dataset.ts_data[ps['f_ts_str']]['vls'][:]
        b = dataset.ts_data[ps['bl_ts_str']]['vls'][:]
        dff = calc_dff(f=f, b=b, background=ps['background'], ep=ps['ep'])

        if return_dff:
            dff_traces[s_id] = dff

        # Get the dff for each event
        s_events = annotations[annotations['subject_id'] == s_id]
        for index in s_events.index:
            event_start = s_events['start'][index]
            event_stop = s_events['end'][index] + 1 # +1 to account for inclusive indexing in table
            extracted_dff[index] = calc_mean_dff(dff, event_start, event_stop, ps['window_type'],
                                                 ps['window_offset'], ps['window_length'])

    # ==================================================================================================================
    # Remove any events where the $\Delta F /F$ window fell outside of the recorded data

    bad_keys = [k for k, vl in extracted_dff.items() if np.all(np.isnan(vl[0]))]
    for key in bad_keys:
        del extracted_dff[key]

    annotations.drop(bad_keys, axis='index', inplace=True)

    # ==================================================================================================================
    # Put $\Delta F/F$ into annotations table

    annotations['dff'] = pd.Series({i:extracted_dff[i][0] for i in extracted_dff.keys()})
    annotations['starts_within_event'] = pd.Series({i:extracted_dff[i][1] for i in extracted_dff.keys()})
    annotations['stops_within_event'] = pd.Series({i:extracted_dff[i][2] for i in extracted_dff.keys()})

    # ==================================================================================================================
    # Enforce using only contained events if we need to
    if ps['enforce_contained_events']:
        keep_events = (annotations['starts_within_event'] == True) & (annotations['stops_within_event'] == True)
        annotations = annotations[keep_events]

    # ==================================================================================================================
    # Now see how many subjects we have for each transition and the total number of transitions as well
    n_trans = count_transitions(annotations, before_str='beh_before', after_str='beh')

    # ==================================================================================================================
    # Get list of preceding and succeeding behaviors we see in enough subjects and events
    if ps['min_n_subjs'] > 1:
        raise(RuntimeError('Support for min_n_subjs not implemented yet.'))

    n_pre_beh_events = n_trans.sum(axis=1)
    n_succ_beh_events = n_trans.sum(axis=0)

    keep_pre_behs = set([i for i in n_pre_beh_events.index if (n_pre_beh_events[i] >= ps['min_n_events'])])
    keep_succ_behs = set([i for i in n_succ_beh_events.index if (n_succ_beh_events[i] >= ps['min_n_events'])])

    # ==================================================================================================================
    # Down select to only those events with preceding and succeeding behaviors that appear enough overall to analyze

    keep_annots = np.asarray([True if ((annotations['beh_before'][i] in keep_pre_behs) and
                                       (annotations['beh'][i] in keep_succ_behs))
                              else False for i in annotations.index])

    analyze_annotations = annotations[keep_annots]

    analyzed_n_subjs_per_trans = count_unique_subjs_per_transition(annotations, before_str='beh_before', after_str='beh')
    analyzed_n_trans = count_transitions(annotations, before_str='beh_before', after_str='beh')

    analyze_trans = [[(bb, ab) for ab in analyzed_n_trans.loc[bb].index if analyzed_n_trans[ab][bb] > 1]
                     for bb in analyzed_n_trans.index]
    analyze_trans = list(itertools.chain(*analyze_trans))

    # ==================================================================================================================
    # Make sure our reference conditions are present
    an_pre_behs = analyze_annotations['beh_before'].unique()
    an_behs = analyze_annotations['beh'].unique()

    if not ps['pre_ref_beh'] in an_pre_behs:
        raise(RuntimeError('The behavior ' + ps['pre_ref_beh'] + ' is not in the analyzed preceding behaviors.'))
    if not ps['ref_beh'] in an_behs:
        raise(RuntimeError('The behavior ' + ps['ref_beh'] + ' is not in the analyzed behaviors.'))

    # ==================================================================================================================
    # Generate our regressors and group indicator variables
    encode_pre_behs = list(set(an_pre_behs) - set(ps['pre_ref_beh']))
    encode_behs = list(set(an_behs) - set(ps['ref_beh']))
    x, mdl_vars = one_hot_from_table(table=analyze_annotations, beh_before=encode_pre_behs, beh_after=encode_behs,
                                     beh_before_str='beh_before', beh_after_str='beh')

    x = np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)
    mdl_vars = mdl_vars + ['ref_' + ps['pre_ref_beh'] + '_' + ps['ref_beh']]

    n_events = len(analyze_annotations)
    unique_ids = analyze_annotations['subject_id'].unique()
    g = np.zeros(n_events)
    for u_i, u_id in enumerate(unique_ids):
        g[analyze_annotations['subject_id'] == u_id] = u_i

   # ==================================================================================================================
    # Now actually calculate our statistics
    dff = np.stack(analyze_annotations['dff'].to_numpy())

    n_analyze_subjs = len(analyze_subjs)
    n_rois = dff.shape[1]
    n_cpu = mp.cpu_count()
    if n_analyze_subjs > 1:
        print('Performing stats for multiple subjects.')
        with mp.Pool(n_cpu) as pool:
            full_stats = pool.starmap(_init_fit_multi_subj_stats_f, [(x, dff[:, r_i], g, ps['alpha']) for r_i in range(n_rois)])

    else:
        print('Performing stats for only one subject.')
        with mp.Pool(n_cpu) as pool:
            full_stats = pool.starmap(_init_fit_single_subj_stats_f, [(x, dff[:, r_i], g, ps['alpha']) for r_i in range(n_rois)])

    # Here we do multiple comparisons corrections
    p_vls = np.stack([s['non_zero_p'] for s in full_stats])
    computed_p_vls = np.stack([s['computed'] for s in full_stats])
    computed_p_vls_matrix = np.tile(computed_p_vls[:, np.newaxis], [1, p_vls.shape[1]])
    corrected_p_vls_by, corrected_p_vls_bon = apply_multiple_comparisons_corrections(p_vls=p_vls,
                                                                                     computed_p_vls=computed_p_vls_matrix)
    for s_i, s in enumerate(full_stats):
        s['non_zero_p_corrected_by'] = corrected_p_vls_by[s_i, :]
        s['non_zero_p_corrected_bon'] = corrected_p_vls_bon[s_i, :]

    # ==================================================================================================================
    # Now we calculate mean for each transition we analyze
    mean_trans_vls = dict()
    for t in analyze_trans:
        t_rows = (analyze_annotations['beh_before'] == t[0]) & (analyze_annotations['beh'] == t[1])
        mean_trans_vls[t] = np.mean(dff[t_rows, :], axis=0)

    # ==================================================================================================================
    # Now save our results

    rs = {'ps': ps, 'full_stats': full_stats, 'beh_trans': analyze_trans, 'var_names': mdl_vars,
          'n_subjs_per_trans': analyzed_n_subjs_per_trans, 'n_trans': analyzed_n_trans, 'mean_trans_vls': mean_trans_vls}

    if ps['save_folder'] is not None:
        save_path = Path(ps['save_folder']) / ps['save_name']
        with open(save_path, 'wb') as f:
            pickle.dump(rs, f)

    # Provide output
    return rs, analyze_annotations, dff_traces


def parallel_test_for_diff_than_mean_vls(ps: dict):
    """ Post-processes fit models, comparing in parallel, if the mean for one transition is different that others.

    See the function test_for_diff_than_mean_vls for details of statistical testing.

    The main purpose of this function is to:

     1) Load results from initial fitting a model

     2) Check if post-processed results comparing means for the initial model fit exist.

     3) If no post-processed results exist, to produce them, using parallel computation to speed up processing.

    Args:

        ps: Parameter dictionary specifying:

            save_folder: The folder with the results to be post-processed are saved as well as the folder in which
            the new post-processed results will be saved

            basic_rs_file: The file with the basic results, which are to be post_processed.

    Returns:

        None.  The post-processed results will be saved in a file in the save_folder with a filename that is
        of the form <basic_rs_file>'_mean_cmp_stats.pkl

    """

    # First, see if post-processed results exist for this file
    save_folder = ps['save_folder']
    save_name = ps['basic_rs_file'].split('.')[0] + '_mean_cmp_stats.pkl'
    save_path = Path(save_folder) / save_name

    if not os.path.exists(save_path):

        # Load basic results
        with open(Path(ps['save_folder']) / ps['basic_rs_file'], 'rb') as f:
            basic_rs = pickle.load(f)

        # Perform stats
        print('Done loading results from: ' + str(ps['basic_rs_file']))
        var_names = basic_rs['var_names']

        n_cpu = mp.cpu_count()
        with mp.Pool(n_cpu) as pool:
            all_mean_stats = pool.starmap(test_for_diff_than_mean_vls,
                                          [(s, var_names, 1e-10, ['beh_before', 'beh']) for s in basic_rs['full_stats']])

        # Here we do multiple comparisons corrections
        p_vls = np.stack([s['eq_mean_p'] for s in all_mean_stats])
        computed_p_vls = np.stack([s['computed'] for s in all_mean_stats]).astype('bool')
        corrected_p_vls_by, corrected_p_vls_bon = apply_multiple_comparisons_corrections(p_vls=p_vls,
                                                                                         computed_p_vls=computed_p_vls)
        for s_i, s in enumerate(all_mean_stats):
            s['eq_mean_p_corrected_by'] = corrected_p_vls_by[s_i, :]
            s['eq_mean_p_corrected_bon'] = corrected_p_vls_bon[s_i, :]

        # Now save our results
        rs = {'ps': ps, 'full_stats': all_mean_stats, 'var_names': var_names, 'n_trans': basic_rs['n_trans'],
              'n_subjs_per_trans': basic_rs['n_subjs_per_trans']}

        with open(save_path, 'wb') as f:
            pickle.dump(rs, f)

        print('Done.  Results saved to: ' + str(save_path))

    else:
        print('Found existing post-processed results. File: ' + str(save_path))


# Helper functions go here

def _init_fit_multi_subj_stats_f(x_i, y_i, g_i, alpha_i, y_std_th=1E-10):

    y_i_std = np.std(y_i)
    if y_i_std > y_std_th:
        beta, acm, n_grps = grouped_linear_regression_ols_estimator(x=x_i, y=y_i, g=g_i)
        stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_grps, alpha=alpha_i)
        stats_computed = True
    else:
        n_vars = x_i.shape[1]
        beta = np.zeros(n_vars)
        beta[:] = np.nan
        acm = np.zeros([n_vars, n_vars])
        acm[:] = np.nan
        stats = {'non_zero_p': np.ones(n_vars)}
        stats['non_zero_p'][:] = np.nan
        n_grps = np.nan
        stats_computed = False

    stats['beta'] = beta
    stats['acm'] = acm
    stats['n_grps'] = n_grps
    stats['computed'] = stats_computed

    return stats


def _init_fit_single_subj_stats_f(x_i, y_i, g_i, alpha_i, y_std_th=1E-10):

    y_i_std = np.std(y_i)
    if y_i_std > y_std_th:
        n_grps = x_i.shape[0]
        beta, acm = linear_regression_ols_estimator(x=x_i, y=y_i)
        stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_grps, alpha=alpha_i)
        stats_computed = True
    else:
        n_vars = x_i.shape[1]
        beta = np.zeros(n_vars)
        beta[:] = np.nan
        acm = np.zeros([n_vars, n_vars])
        acm[:] = np.nan
        stats = {'non_zero_p': np.ones(n_vars)}
        stats['non_zero_p'][:] = np.nan
        n_grps = np.nan
        stats_computed = False

    stats['beta'] = beta
    stats['acm'] = acm
    stats['n_grps'] = n_grps
    stats['computed'] = stats_computed
    return stats
