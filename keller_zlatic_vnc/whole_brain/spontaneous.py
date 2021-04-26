""" Tools for fitting statistical models to spontaneous data. """


import glob
import itertools
import os
from pathlib import Path
import pickle

import multiprocessing as mp
import numpy as np
import pandas as pd

from janelia_core.dataprocessing.dataset import ROIDataset
from janelia_core.stats.regression import linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import grouped_linear_regression_ols_estimator

from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import count_transitions
from keller_zlatic_vnc.data_processing import calc_dff
from keller_zlatic_vnc.data_processing import generate_standard_id_for_full_annots
from keller_zlatic_vnc.data_processing import generate_standard_id_for_volume
from keller_zlatic_vnc.data_processing import get_basic_clean_annotations_from_full
from keller_zlatic_vnc.data_processing import read_full_annotations
from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import test_for_diff_than_mean_vls

def fit_init_models(ps: dict):
    """ Fits initial models to spontaneous activity.

    This function will:

        1) Read in annotation and neural data for multiple subjects.  The user supplies paths to where annotation
        data as well as volume (neural) data can be provided for subjects.  Be default, all subjects for which there is
        annotation data as well as volume data will be analyzed.  (See the exclude_subjs argument for how to modify
        this).

        2) Find "clean" spontaneous events for each subject (see below for how clean events are defined)

        3) Filter events by the behavior that is transitioned into (if requested)

        4) Apply a threshold to determine when an event is preceeded by a "quiet" period

        5) Pool preceeding behaviors into one group if requested (and/or pool preceeding/succeeding left and right
        turns into a single turn behavior)

        6) Calculate Delta F/F for all events in a specified window for each event for all supervoxels (see options
        below for specifying the placement of the window)

        7) Remove any events where the window for Delta F/F fell outside the range of recorded data

        8) Optionally, remove any events where the window for Delta F/F fell outside the start and stop of the behavior
        transitioned into

        9) Remove any transitions for consieration if we don't see enough of them in enough subjects or enough of them
        across all subjects (two seperate criteria are applied)

        10) Fit statistical models on a per-voxel basis.  These statistical models estimate the dff for each transition,
        accounting for correlated errors within subjects (if fitting to more than one subject).

        11) Package the results

    Note: One subject (CW_17-11-03-L6-2) has a different name for its saved volume.  If this subject is included
    in the analysis, this function will make sure the correct volume is used.


    Args:
        ps: A dictionary with the following keys:

            annot_folders: list of folders containing a4 and a9 annotation data

            volume_loc_file: file containing locations to registered volumes

            exlude_subjs: list subjects we do not want to include in the analysis

            q_th: the threshold we use (in number of stacks) to determine when a quiet transition has occured

            dataset_folder: subfolder containing the dataset for each subject

            datast_base_folder: base folder where datasets are stored

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

            pool_preceeding_behaviors: True if we want to pool preceeding behaviors

            pool_preceeding_turns: true if we want to pool preceeding left and right turns into one category (only applies if pool_preceeding_behaviors is false)

            pool_succeeding_turns: true if we want to pool succeeding left and right turns into one category

            clean_event_def: The defintion we use for clean events.  Options are:

                disjoint: "Clean" events are those which have *no* overlap with any other event

                decision: A "clean" events can only intersect at most one other event that starts and ends before it

            behaviors: List the types of behaviors we are interested in analyzing - this is for the behaviors we
            transition into. If None, we don't filter events by behavior

            save_folder: Folder we save events into

            save_name: Name of the file to save results in

    """

    # ==================================================================================================================
    # Define helper functions

    def calc_mean_dff(x, start, stop):

        if ps['window_type'] == 'whole_event':
            take_slice = slice(start, stop)
            starts_within_event = True
            stops_within_event = True
        elif ps['window_type'] == 'start_locked':
            start_offset = start + ps['window_offset']
            stop_offset = start_offset + ps['window_length']
            take_slice = slice(start_offset, stop_offset)
            starts_within_event = ps['window_offset'] >= 0
            stops_within_event = (stop >= stop_offset) and (start <= stop_offset)
        else:
            raise(ValueError('The window_type is not recogonized.'))

        if (take_slice.start < 0) or (take_slice.stop > x.shape[0]):
            mn_vls = np.nan
        else:
            mn_vls = np.mean(x[take_slice, :], axis=0)

        return mn_vls, starts_within_event, stops_within_event

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

    # Produce final list of anntation subjects by intersecting the subjects we have annotations for with those we
    # have volumes before and removing any exlude subjects.
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
        tbl['subject_id'] = s_id
        annotations.append(tbl)

    annotations = [get_basic_clean_annotations_from_full(annot, clean_def=ps['clean_event_def'])
                   for annot in annotations]

    annotations = pd.concat(annotations, ignore_index=True)

    # ==================================================================================================================
    # Filter events by the behavior transitioned into if we are suppose to
    if ps['behaviors'] is not None:
        keep_inds = [i for i in annotations.index if annotations['beh'][i] in ps['behaviors']]
        annotations = annotations.iloc[keep_inds]

    # ==================================================================================================================
    # Now threshold transitions to determine when events were preceeded or succeeded by quiet
    annotations.loc[(annotations['start'] - annotations['beh_before_end']) > ps['q_th'], 'beh_before'] = 'Q'
    annotations.loc[(annotations['beh_after_start'] - annotations['end']) > ps['q_th'], 'beh_after'] = 'Q'
    annotations.drop(['beh_before_start', 'beh_before_end', 'beh_after_start', 'beh_after_end'], axis=1, inplace=True)

    # ==================================================================================================================
    # Pool preceeding behaviors into one (G)rouped label if requested
    if ps['pool_preceeding_behaviors']:
        annotations['beh_before'] = 'G'

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
    # Now we read in the $\frac{\Delta F}{F}$ data for all subjects
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
        f=dataset.ts_data[ps['f_ts_str']]['vls'][:]
        b=dataset.ts_data[ps['bl_ts_str']]['vls'][:]
        dff = calc_dff(f=f, b=b, background=ps['background'], ep=ps['ep'])

        # Get the dff for each event
        s_events = annotations[annotations['subject_id'] == s_id]
        for index in s_events.index:
            event_start = s_events['start'][index]
            event_stop = s_events['end'][index] + 1 # +1 to account for inclusive indexing in table
            extracted_dff[index] = calc_mean_dff(dff, event_start, event_stop)

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
    n_subjs_per_trans = count_unique_subjs_per_transition(annotations, before_str='beh_before', after_str='beh')
    n_trans = count_transitions(annotations, before_str='beh_before', after_str='beh')

    # ==================================================================================================================
    # Get list of transitions we observe in enough subjects to analyze
    analyze_trans = [[(bb, ab) for ab in n_subjs_per_trans.loc[bb].index
                      if (n_subjs_per_trans[ab][bb] >= ps['min_n_subjs'] and n_trans[ab][bb] > ps['min_n_events'])]
                      for bb in n_subjs_per_trans.index]
    analyze_trans = list(itertools.chain(*analyze_trans))

    # ==================================================================================================================
    # Down-select events in annotations to only those with transitions that we will analyze
    keep_codes = [b[0] + b[1] for b in analyze_trans]
    annot_trans_codes = [annotations['beh_before'][i] + annotations['beh'][i] for i in annotations.index]
    keep_annots = np.asarray([True if code in keep_codes else False for code in annot_trans_codes])

    analyze_annotations = annotations[keep_annots]

    # ==================================================================================================================
    # Generate our regressors and group indicator variables
    n_events = len(analyze_annotations)
    n_analyze_trans = len(analyze_trans)

    unique_ids = analyze_annotations['subject_id'].unique()
    g = np.zeros(n_events)
    for u_i, u_id in enumerate(unique_ids):
        g[analyze_annotations['subject_id'] == u_id] = u_i

    x = np.zeros([n_events, n_analyze_trans])
    for row_i in range(n_events):
        event_trans_code = analyze_annotations.iloc[row_i]['beh_before'] + analyze_annotations.iloc[row_i]['beh']
        event_trans_col = np.argwhere(np.asarray(keep_codes) == event_trans_code)[0][0]
        x[row_i, event_trans_col] = 1

    # ==================================================================================================================
    # Now actually calculate our statistics
    dff = np.stack(analyze_annotations['dff'].to_numpy())

    n_analyze_subjs = len(analyze_subjs)
    if n_analyze_subjs > 1:
        print('Performing stats for multiple subjects.')

        def stats_f(x_i, y_i, g_i, alpha_i):
            beta, acm, n_grps = grouped_linear_regression_ols_estimator(x=x_i, y=y_i, g=g_i)
            stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_grps, alpha=alpha_i)
            stats['beta'] = beta
            stats['acm'] = acm
            stats['n_grps'] = n_grps
            return stats
    else:
        print('Performing stats for only one subject.')

        def stats_f(x_i, y_i, g_i, alpha_i):
            n_grps = x_i.shape[0]
            beta, acm = linear_regression_ols_estimator(x=x_i, y=y_i)
            stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_grps, alpha=alpha_i)
            stats['beta'] = beta
            stats['acm'] = acm
            stats['n_grps'] = n_grps
            return stats

    n_rois = dff.shape[1]
    full_stats = [stats_f(x_i=x, y_i=dff[:, r_i], g_i=g, alpha_i=ps['alpha']) for r_i in range(n_rois)]

    # ==================================================================================================================
    # Now save our results

    rs = {'ps': ps, 'full_stats': full_stats, 'beh_trans': analyze_trans}

    save_path = Path(ps['save_folder']) / ps['save_name']
    with open(save_path, 'wb') as f:
        pickle.dump(rs, f)


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
        beh_trans = basic_rs['beh_trans']

        print('Done loading results from: ' + str(ps['basic_rs_file']))

        n_cpu = mp.cpu_count()
        with mp.Pool(n_cpu) as pool:
            all_mean_stats = pool.starmap(test_for_diff_than_mean_vls,
                                          [(s, beh_trans) for s in basic_rs['full_stats']])

        # Now save our results
        rs = {'ps': ps, 'full_stats': all_mean_stats, 'beh_trans': basic_rs['beh_trans']}

        with open(save_path, 'wb') as f:
            pickle.dump(rs, f)

        print('Done.  Results saved to: ' + str(save_path))

    else:
        print('Found existing post-processed results. File: ' + str(save_path))


