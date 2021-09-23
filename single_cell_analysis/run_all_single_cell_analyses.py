""" Runs a batch of single-cell analyses.  """
import copy
import glob
from pathlib import Path
import re

from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from janelia_core.dataprocessing.baseline import percentile_filter_1d
from janelia_core.stats.regression import grouped_linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import grouped_linear_regression_acm_linear_restriction_stats
from janelia_core.stats.regression import visualize_coefficient_stats

from keller_zlatic_vnc.data_processing import calc_dff
from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import find_before_and_after_events
from keller_zlatic_vnc.data_processing import generate_standard_id_for_full_annots
from keller_zlatic_vnc.data_processing import read_full_annotations
from keller_zlatic_vnc.data_processing import read_trace_data
from keller_zlatic_vnc.data_processing import single_cell_extract_dff_with_anotations
from keller_zlatic_vnc.linear_modeling import one_hot_from_table
from keller_zlatic_vnc.utils import form_combinations_from_dict

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

base_ps = dict()

# The file specifying which subjects we should include in the analysis
base_ps['subject_file'] = r'/Volumes/bishoplab/projects/keller_vnc/data/single_cell/subjects.csv'

# Location of files provided by Chen containing the raw fluorescence traces for the single cells
base_ps['trace_base_folder'] = r'/Volumes/bishoplab/projects/keller_vnc/data/single_cell/single_cell_traces'
base_ps['a00c_trace_folder'] = 'A00c'
base_ps['basin_trace_folder'] = 'Basin'
base_ps['handle_trace_folder'] = 'Handle'

# Location of folders containing annotations
base_ps['a4_annot_folder'] = r'/Volumes/bishoplab/projects/keller_vnc/data/full_annotations/behavior_csv_cl_A4'
base_ps['a9_annot_folder'] = r'/Volumes/bishoplab/projects/keller_vnc/data/full_annotations/behavior_csv_cl_A9'

# Specify the type of neurons we analyze
base_ps['cell_type'] = ['a00c', 'handle', 'basin']

# Specfy the cell ids we analyze as a list. If None, we analyze all cell ids
base_ps['cell_ids']  = None

# Specify the manipulation target for subjects we want to analyze, None indicates both A4 and A9
base_ps['man_tgt'] = [None, 'A4', 'A9']

# Say if we should pool preceeding and succeeding turns
base_ps['pool_pre_turns'] = False
base_ps['pool_succ_turns'] = False

# Parameters for declaring preceeding and succeeding quiet behaviors
base_ps['pre_q_th'] = 50
base_ps['succ_q_th'] = 9

# Specify which behaviors we are willing to include in the analysis - b/c we have to see each behavior in
# enough subjects (see below) all of these behaviors may not be included in an analysis, but this gives the
# list of what we are least willing to consider.  If None, all behaviors will be considered
base_ps['behs'] = [['Q', 'TL', 'TR', 'B', 'F', 'H']]

# Specify the reference behavior
base_ps['ref_beh'] = 'Q'

# Specify the minimum number of subjects we have to see preceeding and succeeding behaviors in to include in the
# analysis
base_ps['min_n_pre_subjs'] = 3

# Specify the minimum number of subjects we have to see preceeding and succeeding behaviors in to include in the
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

# The test type we want to peform
base_ps['test_type'] = ['prediction_dependence',
                        'decision_dependence',
                        'state_dependence',
                        'after_reporting',
                        'before_reporting']

# The folder where we should save results
save_loc = '/Volumes/bishoplab/projects/keller_vnc/results/single_cell/new_results/'
save_pdf_name = 'all_tests.pdf'
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
a4_files = np.zeros(n_annot_files, dtype=np.bool)
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
# Get rid of any events where we could not classify the type of preceeding or succeeding behavior
# ======================================================================================================================
subj_events = subj_events.dropna()

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
        ps['dff_window_ref'] = 'end'
        ps['dff_window_offset'] = 1
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
    subj_events_cp.loc[delta_before > ps['pre_q_th'], 'beh_before'] = 'Q'
    subj_events_cp.loc[delta_after > ps['succ_q_th'], 'beh_after'] = 'Q'

    # Down select events based on manipulation target
    if ps['man_tgt'] is not None:
        subj_events_cp = subj_events_cp[subj_events_cp['manipulation_tgt'] == ps['man_tgt']]

    # Pool turns if we are suppose to
    if ps['pool_pre_turns']:
        turn_rows = (subj_events_cp['beh_before'] == 'TL') | (subj_events_cp['beh_before'] == 'TR')
        subj_events_cp.loc[turn_rows, 'beh_before'] = 'TC'

    if ps['pool_succ_turns']:
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

    if not ps['ref_beh'] in set(before_behs):
        raise(RuntimeError('The reference behavior (' + ps['ref_beh'] + ')  is not in the set of preceding behaviors.'))
    if not ps['ref_beh'] in set(after_behs):
        raise (RuntimeError('The reference behavior (' + ps['ref_beh'] + ')  is not in the set of succeeding behaviors.'))

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

    mdl_stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_gprs, alpha=.05)

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

    # Generate output
    visualize_coefficient_stats(var_strs=one_hot_vars_ref, theta=beta, c_ints=mdl_stats['c_ints'],
                                sig=mdl_stats['non_zero'],
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

    pdf.cell(40, 5, 'One vs. Other p-values', ln=1)
    for i, var in enumerate(cmp_vars):
        pdf.cell(40, 5, var + ': ' + '{:.2e}'.format(cmp_p_vls[i]), ln=1)

#  Output the pdf
pdf_save_path = Path(save_loc) / save_pdf_name
pdf.output(pdf_save_path, 'F')

