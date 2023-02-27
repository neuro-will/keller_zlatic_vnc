"""Tools for fitting models to whole brain data."""

import itertools
import multiprocessing as mp
from pathlib import Path
import pickle


import numpy as np

from keller_zlatic_vnc.data_processing import count_transitions
from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.linear_modeling import one_hot_from_table
from keller_zlatic_vnc.whole_brain import spontaneous


def fit_init_models(ps: dict):
    """ A function for fitting initial models to whole-brain closed loop activity.

    This function will:

        1) Load preprocessed data, produced by the notebook dff_extraction.  This notebook takes care of annotating
        quiet events preceding or following each stimulus and extracting DFF for each stimulus event for all ROIs in
        the brain.

        2) Down-select event based on manipulation target.

        3) Optionally pool turns

        4) Applying a filtering to events to only consider those with given preceding or succeeding behaviors.

        5) Drop behaviors that do not appear in enough subjects

        6) Produce summary numbers of the number of subjects each transition type is observed in and total number of
           each type of transition.

        7) Fit models to all ROIs, and perform multiple comparison corrections.

    Args:
        ps: A dictionary with the following fields:
            processed_data_file - File with extracted DFF for all rois and events across subjects.
            manipulation_tgt - The manipulation targer we want to analyze events for.  Either 'A4', 'A9' or
                None (which indicates pooling).
            pool_turns - True if we are suppose to pool left and right turns
            behs - The set of before and after behaviors we want to fit models to.  If None, all behaviors will be used.
            min_n_pre_subjs - Minimum number of subjects we must see a preceding behavior in to include in the analysis
            min_n_succ_subjs - Minimum number of subjects we must see a succeeding behavior in to include in the analysis
            ref_beh - The reference behavior for modeling.
            ind_alpha - The significance level we reject individual null hypotheses at at
    """

    # ==================================================================================================================
    # Load the processed data
    with open(ps['processed_data_file'], 'rb') as f:
        processed_data = pickle.load(f)
        subject_event_data = processed_data['subject_event_data']

    # Down select events based on manipulation target
    if ps['manipulation_tgt'] is not None:
        subject_event_data = subject_event_data[subject_event_data['manipulation_tgt'] == ps['manipulation_tgt']]

    # Pool turns if we are suppose to
    if ps['pool_turns']:
        turn_rows = (subject_event_data['beh_before'] == 'TL') | (subject_event_data['beh_before'] == 'TR')
        subject_event_data.loc[turn_rows, 'beh_before'] = 'TC'

        turn_rows = (subject_event_data['beh_after'] == 'TL') | (subject_event_data['beh_after'] == 'TR')
        subject_event_data.loc[turn_rows, 'beh_after'] = 'TC'

    # Down select to only the type of behaviors we are willing to consider
    if ps['behs'] is not None:
        keep_inds = [i for i in subject_event_data.index if subject_event_data['beh_before'][i] in set(ps['behs'])]
        subject_event_data = subject_event_data.loc[keep_inds]

        keep_inds = [i for i in subject_event_data.index if subject_event_data['beh_after'][i] in set(ps['behs'])]
        subject_event_data = subject_event_data.loc[keep_inds]

    # Drop any behaviors that do not appear in enough subjects
    subj_trans_counts = count_unique_subjs_per_transition(table=subject_event_data)
    n_before_subjs = subj_trans_counts.sum(axis=1)
    n_after_subjs = subj_trans_counts.sum(axis=0)

    before_an_behs = set([i for i in n_before_subjs.index if n_before_subjs[i] >= ps['min_n_pre_subjs']])
    after_an_behs = set([i for i in n_after_subjs.index if n_after_subjs[i] >= ps['min_n_succ_subjs']])

    keep_inds = [i for i in subject_event_data.index if subject_event_data['beh_before'][i] in before_an_behs]
    subject_event_data = subject_event_data.loc[keep_inds]

    keep_inds = [i for i in subject_event_data.index if subject_event_data['beh_after'][i] in after_an_behs]
    subject_event_data = subject_event_data.loc[keep_inds]

    # ==================================================================================================================
    # Get summary statistics on transitions analyzed in the analysis
    analyzed_n_subjs_per_trans = count_unique_subjs_per_transition(subject_event_data, before_str='beh_before',
                                                                   after_str='beh_after')
    analyzed_n_trans = count_transitions(subject_event_data, before_str='beh_before', after_str='beh_after')

    analyze_trans = [[(bb, ab) for ab in analyzed_n_trans.loc[bb].index if analyzed_n_trans[ab][bb] > 1]
                     for bb in analyzed_n_trans.index]
    analyze_trans = list(itertools.chain(*analyze_trans))

    # ==================================================================================================================
    # Prepare matrices of data

    # Find grouping of data by subject
    unique_ids = subject_event_data['subject_id'].unique()
    g = np.zeros(len(subject_event_data))
    for u_i, u_id in enumerate(unique_ids):
        g[subject_event_data['subject_id'] == u_id] = u_i

    # Generate representation of behaviors for model fitting
    before_behs = subject_event_data['beh_before'].unique()
    after_behs = subject_event_data['beh_after'].unique()

    before_behs_ref = list(set(before_behs).difference(ps['ref_beh']))
    after_behs_ref = list(set(after_behs).difference(ps['ref_beh']))

    one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(subject_event_data, beh_before=before_behs_ref,
                                                            beh_after=after_behs_ref)

    one_hot_data_ref = np.concatenate([one_hot_data_ref, np.ones([one_hot_data_ref.shape[0], 1])], axis=1)
    one_hot_vars_ref = one_hot_vars_ref + ['ref']

    _, v, _ = np.linalg.svd(one_hot_data_ref)
    if np.min(v) < .001:
        raise (RuntimeError('regressors are nearly co-linear'))

    # ==================================================================================================================
    # Fit models to each ROI and perform statistics
    dff = np.stack(subject_event_data['dff'].to_numpy())

    n_rois = dff.shape[1]
    n_cpu = mp.cpu_count()
    with mp.Pool(n_cpu) as pool:
        full_stats = pool.starmap(spontaneous._init_fit_multi_subj_stats_f, [(one_hot_data_ref,
                                                                              dff[:, r_i],
                                                                              g,
                                                                              ps['ind_alpha'])
                                                                             for r_i in range(n_rois)]
                                  )
    # Here we do multiple comparisons corrections
    p_vls = np.stack([s['non_zero_p'] for s in full_stats])
    computed_p_vls = np.stack([s['computed'] for s in full_stats])
    computed_p_vls_matrix = np.tile(computed_p_vls[:, np.newaxis], [1, p_vls.shape[1]])
    corrected_p_vls_by, corrected_p_vls_bon = spontaneous.apply_multiple_comparisons_corrections(
        p_vls=p_vls,
        computed_p_vls=computed_p_vls_matrix
    )

    for s_i, s in enumerate(full_stats):
        s['non_zero_p_corrected_by'] = corrected_p_vls_by[s_i, :]
        s['non_zero_p_corrected_bon'] = corrected_p_vls_bon[s_i, :]

    # ==================================================================================================================
    # Now we calculate mean for each transition we analyze
    mean_trans_vls = dict()
    for t in analyze_trans:
        t_rows = (subject_event_data['beh_before'] == t[0]) & (subject_event_data['beh_after'] == t[1])
        mean_trans_vls[t] = np.mean(dff[t_rows, :], axis=0)

    # ==================================================================================================================
    # Now save our results
    rs = {'ps': ps, 'full_stats': full_stats, 'beh_trans': analyze_trans, 'var_names': one_hot_vars_ref,
          'n_subjs_per_trans': analyzed_n_subjs_per_trans, 'n_trans': analyzed_n_trans,
          'mean_trans_vls': mean_trans_vls}

    save_path = Path(ps['save_folder']) / ps['save_name']
    with open(save_path, 'wb') as f:
        pickle.dump(rs, f)
