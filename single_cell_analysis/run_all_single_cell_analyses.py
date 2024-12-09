""" Runs a batch of single-cell analyses.  """
import copy
import glob
from pathlib import Path
import re

from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from janelia_core.dataprocessing.baseline import percentile_filter_1d
from janelia_core.stats.multiple_comparisons import apply_bonferroni
from janelia_core.stats.regression import grouped_linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import grouped_linear_regression_acm_linear_restriction_stats
from janelia_core.stats.regression import visualize_coefficient_stats

from keller_zlatic_vnc.data_processing import calc_dff
from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import down_select_events
from keller_zlatic_vnc.data_processing import find_before_and_after_events
from keller_zlatic_vnc.data_processing import generate_standard_id_for_full_annots
from keller_zlatic_vnc.data_processing import read_full_annotations
from keller_zlatic_vnc.data_processing import read_raw_transitions_from_excel
from keller_zlatic_vnc.data_processing import read_trace_data
from keller_zlatic_vnc.data_processing import single_cell_extract_dff_with_anotations
from keller_zlatic_vnc.linear_modeling import one_hot_from_table
from keller_zlatic_vnc.utils import form_combinations_from_dict

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

base_ps = dict()

# The file specifying which subjects we should include in the analysis
base_ps['subject_file'] = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\single_cell\subjects.csv'
# Location of files provided by Chen containing the raw fluorescence traces for the single cells
base_ps['trace_base_folder'] = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\single_cell\single_cell_traces'
base_ps['a00c_trace_folder'] = 'A00c'
base_ps['basin_trace_folder'] = 'Basin'
base_ps['handle_trace_folder'] = 'Handle'

# Location of folders containing annotations
base_ps['a4_annot_folder'] = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\full_annotations/behavior_csv_cl_A4'
base_ps['a9_annot_folder'] = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\full_annotations/behavior_csv_cl_A9'

# Location of file containing Chen's annotations - we use this to filter down to only good stimulus events
base_ps['chen_file'] = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\extracted_dff_v2\transition_list_CW_11202021.xlsx'

# Specify the type of neurons we analyze
base_ps['cell_type'] = ['a00c']  # 'a00c' 'handle', 'basin'

# Specify the cell ids we analyze as a list. If None, we analyze all cell ids

# a00c
if base_ps['cell_type'][0] == 'a00c':
    base_ps['cell_ids'] = [None,
                           ['ds', 'antL', 'antR'],
                           ['ds', 'midL', 'midR'],
                           ['ds', 'postL', 'postR']]

# basin
if base_ps['cell_type'][0] == 'basin':
    base_ps['cell_ids'] = [None,
                           ['ds', 'A1R', 'A1L', '1AL', '1AR'],
                           ['ds', 'A2L', 'A2R', '2AL', '2AR'],
                           ['ds', 'A3R', 'A3L', '3AL', '3AR'],
                           ['ds', 'A4R', 'A4L', '4AL', '4AR'],
                           ['ds', 'A5L','A5R', '5AL', '5AR', 'A5L '],
                           ['ds', 'A6L','A6R', '6AL', '6AR'],
                           ['ds', 'A7L', 'A7R', '7AL', '7AR'],
                           ['ds', 'A8L', 'A8R', '8AL', '8AR'],
                           ['ds', 'A9R', 'A9L', '9AL', '9AR'],
                           ['ds', 'T1L', '1TL', 'T1R'],
                           ['ds', 'T2R', 'T2L', '2TL', '2TR'],
                           ['ds', 'T3L', 'T3R', '3TL', '3TR']]

# handle
if base_ps['cell_type'][0] == 'handle':
    base_ps['cell_ids'] = [None,
                           ['ds', 'A1'],
                           ['ds', 'A2'],
                           ['ds', 'A3'],
                           ['ds', 'A4'],
                           ['ds', 'A5', 'A5  '],
                           ['ds', 'A6'],
                           ['ds', 'A7'],
                           ['ds', 'A8'],
                           ['ds', 'A9'],
                           ['ds', 'T1'],
                           ['ds', 'T2'],
                           ['ds', 'T3'],
                           ['ds', 'SEG']]

# Specify the manipulation target for subjects we want to analyze, None indicates both A4 and A9
base_ps['man_tgt'] = [None] #[None, 'A4', 'A9']

# Say if we should pool preceeding and succeeding turns
base_ps['pool_turns'] = [True] #[True, False]

# Parameters for declaring preceeding and succeeding quiet behaviors
base_ps['pre_q_th'] = 50
base_ps['succ_q_th'] = 9

# Parameters for determining the location of the marked preceding and succeeding quiet events
base_ps['pre_q_event_l'] = 3 # Event length for the preceding quiet event
base_ps['succ_q_event_l'] = 3 # Event length for the succeeding quiet event

# Specify which behaviors we are willing to include in the analysis - b/c we have to see each behavior in
# enough subjects (see below) all of these behaviors may not be included in an analysis, but this gives the
# list of what we are least willing to consider.  If None, all behaviors will be considered
base_ps['behs'] = ['ds', 'Q', 'TC', 'TL', 'TR',  'B', 'F', 'H']

# Specify the reference behavior
base_ps['ref_beh'] = 'Q'

# Specify the minimum number of subjects we have to see preceding and succeeding behaviors in to include in the
# analysis
base_ps['min_n_pre_subjs'] = 3

# Specify the minimum number of subjects we have to see preceding and succeeding behaviors in to include in the
# analysis
base_ps['min_n_succ_subjs'] = 3

# Parameters for calculating Delta F/F

base_ps['baseline_calc_params'] = dict()
base_ps['baseline_calc_params']['window_length'] = 30001
base_ps['baseline_calc_params']['filter_start'] = -1500
base_ps['baseline_calc_params']['write_offset'] = 1500
base_ps['baseline_calc_params']['p'] = 0.1

base_ps['dff_calc_params'] = dict()
base_ps['dff_calc_params']['background'] = 100
base_ps['dff_calc_params']['ep'] = 20

# The test type we want to perform
base_ps['test_type'] = ['prediction_dependence',
                        'decision_dependence',
                        'state_dependence',
                        'after_reporting',
                        'before_reporting']

# The significance level we reject individual null hypotheses at at
base_ps['ind_alpha'] = .05

# The significance level we want to test at, after correcting for multiple comparisons
base_ps['mc_alpha'] = .05

# The folder where we should save results
save_loc = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\single_cell\12_24_single_cell_traces\run_1'
save_pdf_name = 'all_tests.pdf'

# ======================================================================================================================
# Do some basic checks here
# ======================================================================================================================

if base_ps['succ_q_event_l'] > np.floor(base_ps['succ_q_th'] / 2):
    raise (RuntimeError('succ_q_event_l must be less than floor(succ_q_th)/2)'))
if base_ps['pre_q_event_l'] > np.floor(base_ps['pre_q_th'] / 2):
    raise (RuntimeError('pre_q_event_l must be less than floor(pre_q_th)/2)'))

# ======================================================================================================================
# Generate parameter combinations
# ======================================================================================================================
ps_combs = form_combinations_from_dict(base_ps)

# ======================================================================================================================
# Read in the basic data for each subject
# ======================================================================================================================

# Get the list of all subjects we need to process
subjects = list(pd.read_csv(base_ps['subject_file'])['Subject'])

print('===============================================================================================================')
print('Reading in activity data for all subjects.')
print('===============================================================================================================')

data = read_trace_data(subjects=subjects,
                       a00c_trace_folder=Path(base_ps['trace_base_folder'])/base_ps['a00c_trace_folder'],
                       handle_trace_folder=Path(base_ps['trace_base_folder'])/base_ps['handle_trace_folder'],
                       basin_trace_folder=Path(base_ps['trace_base_folder'])/base_ps['basin_trace_folder'])

# ======================================================================================================================
# Calculate Delta F/F for each cell
# ======================================================================================================================

print('===============================================================================================================')
print('Calculating Delta F/F for each cell.')
print('===============================================================================================================')

n_cells = data.shape[0]
dff = [None]*n_cells
for cell_row, cell_idx in enumerate(data.index):
    baseline = percentile_filter_1d(data['f'][cell_idx], **base_ps['baseline_calc_params'])
    dff[cell_row] = calc_dff(f=data['f'][cell_idx], b=baseline, **base_ps['dff_calc_params'])

data['dff'] = dff

# ======================================================================================================================
# Find stimulus events for all subjects
# ======================================================================================================================

print('===============================================================================================================')
print('Reading in stimulus events for all subjects.')
print('===============================================================================================================')

# Get list of subjects we have annotations for
a4_file_paths = glob.glob(str(Path(base_ps['a4_annot_folder']) / '*.csv'))
a9_file_paths = glob.glob(str(Path(base_ps['a9_annot_folder']) / '*.csv'))

n_annot_files = len(a4_file_paths) + len(a9_file_paths)
a4_files = np.zeros(n_annot_files, dtype=bool)
a4_files[0:len(a4_file_paths)] = True

annot_file_paths = a4_file_paths + a9_file_paths

annot_file_names = [Path(p).name for p in annot_file_paths]
annot_subjs = [generate_standard_id_for_full_annots(fn) for fn in annot_file_names]

# Get stimulus events for each subject we analyze
subj_events = pd.DataFrame()

for subj in list(data['subject_id'].unique()):

    # Find the annotations for this subject
    ind = np.argwhere(np.asarray(annot_subjs) == subj)
    if len(ind) == 0:
        raise (RuntimeError('Unable to find annotations for subject ' + subj + '.'))
    else:
        ind = ind[0][0]

    # Load the annotations for this subject
    tbl = read_full_annotations(annot_file_paths[ind])

    # Pull out stimulus events for this subject, noting what comes before and after
    stim_tbl = copy.deepcopy(tbl[tbl['beh'] == 'S'])
    stim_tbl.insert(0, 'subject_id', subj)
    stim_tbl.insert(1, 'event_id', range(stim_tbl.shape[0]))
    if a4_files[ind] == True:
        stim_tbl.insert(2, 'manipulation_tgt', 'A4')
    else:
        stim_tbl.insert(2, 'manipulation_tgt', 'A9')
    before_after_tbl = find_before_and_after_events(events=stim_tbl, all_events=tbl)
    stim_annots = pd.concat([stim_tbl, before_after_tbl], axis=1)
    subj_events = subj_events.append(stim_annots, ignore_index=True)

# ======================================================================================================================
# Get rid of any events where we could not classify the type of preceding or succeeding behavior
# ======================================================================================================================
subj_events = subj_events.dropna()

# ======================================================================================================================
# Get rid of any stimulus events which are not also in Chen's annotations - we do this because some of
# the stimulus events in the full annotations (Nadine's annotations) should be removed because of artefacts. Chen's
# annotations only include the stimulus events we should analyze.
# ======================================================================================================================
chen_events = read_raw_transitions_from_excel(file=base_ps['chen_file'])
chen_events = chen_events.rename(columns={'Manipulation Start': 'start', 'Manipulation End': 'end'})
subj_events = down_select_events(tbl_1=subj_events, tbl_2=chen_events)

# ======================================================================================================================
# Now process all results
# ======================================================================================================================
n_ps_combs = len(ps_combs)

pdf = FPDF('L')

# Set font for producing plots
plt.rc('font', family='arial', weight='normal', size=15)

for ps_i, ps in enumerate(ps_combs):

    if ps['test_type'] == 'after_reporting':
        # The type of window we are aligning
        ps['dff_window_type'] = 'start_aligned'

        # The reference we use for aligning windows
        ps['dff_window_ref'] = 'beh_after_start'

        # The offset we applying when placing DFF windows
        ps['dff_window_offset'] = 0

        # The length of the window we calculate DFF in
        ps['dff_window_length'] = 3

        # The event we align the end of the window to (if not using windows of fixed length)
        ps['dff_window_end_ref'] = None

        # The offset we use when locating the end of the window (if not using windows of fixed length)
        ps['dff_window_end_offset'] = None

    elif ps['test_type'] == 'before_reporting':
        ps['dff_window_type'] = 'start_aligned'
        ps['dff_window_ref'] = 'beh_before_start'
        ps['dff_window_offset'] = 0
        ps['dff_window_length'] = 3
        ps['dff_window_end_ref'] = None
        ps['dff_window_end_offset'] = None
    elif ps['test_type'] == 'prediction_dependence':
        ps['dff_window_type'] = 'start_aligned'
        ps['dff_window_ref'] = 'start'
        ps['dff_window_offset'] = -3
        ps['dff_window_length'] = 3
        ps['dff_window_end_ref'] = None
        ps['dff_window_end_offset'] = None
    elif ps['test_type'] == 'state_dependence':
        ps['dff_window_type'] = 'start_aligned'
        ps['dff_window_ref'] = 'beh_after_start'
        ps['dff_window_offset'] = 0
        ps['dff_window_length'] = 3
        ps['dff_window_end_ref'] = None
        ps['dff_window_end_offset'] = None
    elif ps['test_type'] == 'decision_dependence':
        ps['dff_window_type'] = 'start_end_aligned'
        ps['dff_window_ref'] = 'start'
        ps['dff_window_offset'] = 0
        ps['dff_window_length'] = None
        ps['dff_window_end_ref'] = 'end'
        ps['dff_window_end_offset'] = 1 # +1 to include the last point of stimulation
    else:
        raise(ValueError('The test type ' + ps['test_type'] + ' is not recognized.'))

    print('===============================================================================================================')
    print('Performing analysis for parameter values ' + str(ps_i + 1) + ' of ' + str(n_ps_combs) + '.')
    print(ps['test_type'])
    print('===============================================================================================================')

    # Make copies of data and annotations, which we will modify
    data_cp = copy.deepcopy(data)
    subj_events_cp = copy.deepcopy(subj_events)

    # Down select to the cell types we want to analyze
    data_cp = data_cp[data_cp['cell_type'] == ps['cell_type']]

    # Down select by cell id
    if ps['cell_ids'] is not None:
        data_cp = data_cp[data_cp['cell_id'].isin(ps['cell_ids'])]

    # Mark preceeding and succeeding quiet events
    delta_before = subj_events_cp['start'] - subj_events_cp['beh_before_end']
    delta_after = subj_events_cp['beh_after_start'] - subj_events_cp['end']

    before_quiet_inds = delta_before > ps['pre_q_th']
    after_quiet_inds = delta_after > ps['succ_q_th']

    subj_events_cp.loc[before_quiet_inds, 'beh_before'] = 'Q'
    subj_events_cp.loc[after_quiet_inds, 'beh_after'] = 'Q'

    # Mark the start and stop of the marked quiet events
    new_before_start = (np.ceil((subj_events_cp[before_quiet_inds]['start'] -
                                subj_events_cp[before_quiet_inds]['beh_before_end']) / 2) +
                       subj_events_cp[before_quiet_inds]['beh_before_end'])
    new_before_end = new_before_start + ps['pre_q_event_l'] - 1  # Minus 1 for inclusive indexing
    subj_events_cp.loc[before_quiet_inds, 'beh_before_start'] = new_before_start
    subj_events_cp.loc[before_quiet_inds, 'beh_before_end'] = new_before_end

    new_after_start = (np.ceil((subj_events_cp[after_quiet_inds]['beh_after_start'] -
                                subj_events_cp[after_quiet_inds]['end']) / 2) +
                       subj_events_cp[after_quiet_inds]['end'])
    new_after_end = new_after_start + ps['succ_q_event_l'] - 1  # Minus 1 for inclusive indexing
    subj_events_cp.loc[after_quiet_inds, 'beh_after_start'] = new_after_start
    subj_events_cp.loc[after_quiet_inds, 'beh_after_end'] = new_after_end

    # Down select events based on manipulation target
    if ps['man_tgt'] is not None:
        subj_events_cp = subj_events_cp[subj_events_cp['manipulation_tgt'] == ps['man_tgt']]

    # Pool turns if we are suppose to
    if ps['pool_turns']:
        turn_rows = (subj_events_cp['beh_before'] == 'TL') | (subj_events_cp['beh_before'] == 'TR')
        subj_events_cp.loc[turn_rows, 'beh_before'] = 'TC'

    if ps['pool_turns']:
        turn_rows = (subj_events_cp['beh_after'] == 'TL') | (subj_events_cp['beh_after'] == 'TR')
        subj_events_cp.loc[turn_rows, 'beh_after'] = 'TC'

    # Down select to only the type of behaviors we are willing to consider
    if ps['behs'] is not None:
        keep_inds = [i for i in subj_events_cp.index if subj_events_cp['beh_before'][i] in set(ps['behs'])]
        subj_events_cp = subj_events_cp.loc[keep_inds]

        keep_inds = [i for i in subj_events_cp.index if subj_events_cp['beh_after'][i] in set(ps['behs'])]
        subj_events_cp = subj_events_cp.loc[keep_inds]

    # Drop any behaviors that do not appear in enough subjects
    subj_trans_counts = count_unique_subjs_per_transition(table=subj_events_cp)
    n_before_subjs = subj_trans_counts.sum(axis=1)
    n_after_subjs = subj_trans_counts.sum(axis=0)

    before_an_behs = set([i for i in n_before_subjs.index if n_before_subjs[i] >= ps['min_n_pre_subjs']])
    after_an_behs = set([i for i in n_after_subjs.index if n_after_subjs[i] >= ps['min_n_succ_subjs']])

    keep_inds = [i for i in subj_events_cp.index if subj_events_cp['beh_before'][i] in before_an_behs]
    subj_events_cp = subj_events_cp.loc[keep_inds]

    keep_inds = [i for i in subj_events_cp.index if subj_events_cp['beh_after'][i] in after_an_behs]
    subj_events_cp = subj_events_cp.loc[keep_inds]

    # Pull out Delta F/F for each event and cell along with all information we need for performing statistics
    full_tbl = single_cell_extract_dff_with_anotations(activity_tbl=data_cp, event_tbl=subj_events_cp,
                                                       align_col=ps['dff_window_ref'],
                                                       ref_offset=ps['dff_window_offset'],
                                                       window_l=ps['dff_window_length'],
                                                       window_type=ps['dff_window_type'],
                                                       end_align_col=ps['dff_window_end_ref'],
                                                       end_ref_offset=ps['dff_window_end_offset'])

    # Find grouping of data by subject
    unique_ids = full_tbl['subject_id'].unique()
    g = np.zeros(len(full_tbl))
    for u_i, u_id in enumerate(unique_ids):
        g[full_tbl['subject_id'] == u_id] = u_i

    # Fit initial models and calculate initial statistics
    before_behs = full_tbl['beh_before'].unique()
    after_behs = full_tbl['beh_after'].unique()

    try:
        if not ps['ref_beh'] in set(before_behs):
            raise(RuntimeError('The reference behavior (' + ps['ref_beh'] + ')  is not in the set of preceding behaviors. Skipping analysis'))
        if not ps['ref_beh'] in set(after_behs):
            raise(RuntimeError('The reference behavior (' + ps['ref_beh'] + ')  is not in the set of preceding behaviors. Skipping analysis'))

        before_behs_ref = list(set(before_behs).difference(ps['ref_beh']))
        after_behs_ref = list(set(after_behs).difference(ps['ref_beh']))

        one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(full_tbl, beh_before=before_behs_ref,
                                                                beh_after=after_behs_ref)

        one_hot_data_ref = np.concatenate([one_hot_data_ref, np.ones([one_hot_data_ref.shape[0], 1])], axis=1)
        one_hot_vars_ref = one_hot_vars_ref + ['ref']

        _, v, _ = np.linalg.svd(one_hot_data_ref)
        if np.min(v) < .001:
            raise (RuntimeError('regressors are nearly co-linear'))

        beta, acm, n_gprs = grouped_linear_regression_ols_estimator(x=one_hot_data_ref, y=full_tbl['dff'].to_numpy(), g=g)

        mdl_stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_gprs, alpha=ps['ind_alpha'])

        # Do comparisons of coefficients
        n_grps = len(np.unique(g))

        cmp_vars = one_hot_vars_ref[0:-1]
        cmp_p_vls = np.zeros(len(cmp_vars))

        before_inds = np.asarray([True if re.match('beh_before*', var) else False for var in one_hot_vars_ref])
        after_inds = np.asarray([True if re.match('beh_after*', var) else False for var in one_hot_vars_ref])

        n_before_vars = np.sum(before_inds)
        n_after_vars = np.sum(after_inds)

        for v_i, var in enumerate(cmp_vars):
            if before_inds[v_i] == True:
                cmp_beta = beta[before_inds]
                cmp_acm = acm[np.ix_(before_inds, before_inds)]
                cmp_i = v_i
            else:
                cmp_beta = beta[after_inds]
                cmp_acm = acm[np.ix_(after_inds, after_inds)]
                cmp_i = v_i - n_before_vars

            r = np.ones(len(cmp_beta)) / (len(cmp_beta) - 1)
            r[cmp_i] = -1
            cmp_p_vls[v_i] = grouped_linear_regression_acm_linear_restriction_stats(beta=cmp_beta, acm=cmp_acm, r=r,
                                                                                    q=np.asarray([0]), n_grps=n_grps)

        # Save results
        rs = dict()
        rs['before_behs'] = before_behs
        rs['after_behs'] = after_behs
        rs['one_hot_vars_ref'] = one_hot_vars_ref
        rs['init_fit'] = {'beta': beta, 'acm': acm, 'n_grps': n_grps}
        rs['init_fit_stats'] = mdl_stats
        rs['cmp_stats'] = {'cmp_vars': cmp_vars, 'cmp_p_vls': cmp_p_vls}
        rs['full_tbl'] = full_tbl

        save_file_name = 'rs_' + str(ps_i) + '.pkl'
        with open(Path(save_loc) / save_file_name, 'wb') as f:
            pickle.dump({'ps': ps, 'rs': rs}, f)

        # Generate output
        mdl_bonferroni_sig, _ = apply_bonferroni(p_vls=mdl_stats['non_zero_p'], alpha=ps['mc_alpha'])
        _, cmp_p_vls_bonferroni = apply_bonferroni(p_vls=cmp_p_vls, alpha=ps['mc_alpha'])
        vis_clrs = np.zeros([len(beta), 3])
        vis_clrs[mdl_bonferroni_sig,0] = 1
        visualize_coefficient_stats(var_strs=one_hot_vars_ref, theta=beta, c_ints=mdl_stats['c_ints'],
                                    sig=mdl_bonferroni_sig, var_clrs=vis_clrs,
                                    x_axis_rot=90)
        plt.ylabel('$\Delta F / F$')
        plt.xlabel('Behavior')
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        fig_name = 'fig_' + str(ps_i) + '.png'
        fig_path = Path(save_loc) / fig_name
        plt.savefig(fig_path)
        plt.close()

        # Add page to the pdf for these results
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 7, ps['cell_type'], ln=1)
        pdf.set_font('Arial', size=10)
        if ps['cell_ids'] is not None:
            cell_id_str = ','.join(ps['cell_ids'])
            pdf.cell(40, 5, 'Cell IDs: ' + cell_id_str, ln=1)
        else:
            pdf.cell(40, 5, 'Cell IDs: All', ln=1)
        pdf.cell(40, 5, 'Test type: ' + ps['test_type'], ln=1)
        pdf.cell(40, 5, 'Preceding quiet threshold: ' + str(ps['pre_q_th']), ln=1)
        pdf.cell(40, 5, 'Succeeding quiet threshold: ' + str(ps['succ_q_th']), ln=1)
        pdf.cell(40, 5, 'Min # subjects per preceding behavior: ' + str(ps['min_n_pre_subjs']), ln=1)
        pdf.cell(40, 5, 'Min # subjects per suceeding behavior: ' + str(ps['min_n_succ_subjs']), ln=1)
        if ps['man_tgt'] is not None:
            pdf.cell(40, 5, 'Manipulation target(s): ' + ps['man_tgt'], ln=1)
        else:
            pdf.cell(40, 5, 'Manipulation target(s): ' + 'A4 + A9', ln=1)
        #pdf.ln(10)
        pdf.image(str(fig_path), w=120)

        pdf.cell(40, 5, 'One vs. Other p-values (w/o and w/ Bonferroni)', ln=1)
        for i, var in enumerate(cmp_vars):
            if cmp_p_vls_bonferroni[i] <= ps['mc_alpha']:
                pdf.set_text_color(255, 0, 0)
            else:
                pdf.set_text_color(0, 1, 1)
            pdf.cell(40, 5, var + ': ' + '{:.2e}'.format(cmp_p_vls[i]) + ', {:2e}'.format(cmp_p_vls_bonferroni[i]), ln=1)
        pdf.set_text_color(0, 1, 1)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as err:
        print('Error detected. Skipping this analysis.')


#  Output the pdf
pdf_save_path = Path(save_loc) / save_pdf_name
pdf.output(pdf_save_path, 'F')

