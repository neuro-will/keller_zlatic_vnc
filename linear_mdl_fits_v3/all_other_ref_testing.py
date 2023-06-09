""" Tests for state, decision or reporting dependence in DFF, referencing each behavior to an 'other' condition.

    This script will looking across multiple different ways of processing data.

    """

import pathlib

from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

from keller_zlatic_vnc.data_processing import count_unique_subjs_per_transition
from keller_zlatic_vnc.data_processing import extract_transitions
from keller_zlatic_vnc.data_processing import generate_transition_dff_table
from keller_zlatic_vnc.data_processing import read_raw_transitions_from_excel
from keller_zlatic_vnc.data_processing import recode_beh
from keller_zlatic_vnc.data_processing import A00C_SEG_CODES, BASIN_SEG_CODES, HANDLE_SEG_CODES
from keller_zlatic_vnc.linear_modeling import one_hot_from_table

from janelia_core.stats.regression import grouped_linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import visualize_coefficient_stats

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

ps = dict()

# ======================================================================================================================
# Here we specify where the pdf should be saved and its name in a single path
pdf_save_path = '/Volumes/bishoplab/projects/keller_vnc/results/single_cell/old_results/decision_dep_other_ref.pdf'


# ======================================================================================================================
# Here we specify a path to a folder we can use for saving temporary files (images)
temp_folder = '/Users/bishopw/Desktop/temp'

# ======================================================================================================================
# Here we specify the location of the data

data_folder = r'/Volumes/bishoplab/projects/keller_vnc/data/extracted_dff_v2'
transition_file = 'transition_list.xlsx'

a00c_a4_act_data_file = 'A00c_activity_A4.mat'
a00c_a9_act_data_file = 'A00c_activity_A9.mat'

basin_a4_act_data_file = 'Basin_activity_A4.mat'
basin_a9_act_data_file = 'Basin_activity_A9.mat'

handle_a4_act_data_file = 'Handle_activity_A4.mat'
handle_a9_act_data_file = 'Handle_activity_A9.mat'

# ======================================================================================================================
# Here we specify the type of testing we will do.  Options are:
#
#   state_dependence - tests if dff after manipulation is sensitive to behavior before
#   prediction_dependence - tests if dff before manipulation is sensitive to behavior after
#   decision_dependence - tests if dff during manipulation is sensitive to behavior after
#   before_reporting - tests if dff before manipulation is sensitive to behavior before
#   after_reporting - tests if dff after manipulation is sensitive to behavior after
#
test_type = 'decision_dependence'

# ======================================================================================================================
# Here, we specify the different ways we want to filter the data when fitting models.  We will produce results for all
# possible combinations of the options below.

# Cell types are tuples of form (cell type, list of cell ids).  In place of a list of cell ids, the string 'all'
# indicates we are using all cell ids
cell_types = [('a00c', 'all'),
              ('handle', 'all'),
              ('basin', 'all')]#,
              #('basin', [7]),
              #('basin', [12])]

if False:
    cell_types = [('a00c', [1, 2]),
              ('a00c', [3, 4]),
              ('a00c', [5, 6]),
              ('handle', [1]),
              ('handle', [2]),
              ('handle', [3]),
              ('handle', [4]),
              ('handle', [5]),
              ('handle', [6]),
              ('handle', [7]),
              ('handle', [8]),
              ('handle', [9]),
              ('handle', [10]),
              ('handle', [11]),
              ('handle', [12]),
              ('basin', [1]),
              ('basin', [2]),
              ('basin', [3]),
              ('basin', [4]),
              ('basin', [5]),
              ('basin', [6]),
              ('basin', [7]),
              ('basin', [8]),
              ('basin', [9]),
              ('basin', [10]),
              ('basin', [11]),
              ('basin', [12])]

manip_types = ['A4', 'A9', 'A4+A9']
cut_off_times = [3.231] #, 17.4523, 111.6582]


# Min number of subjects which must display a test behavior to include it in testing
min_n_subjects_per_beh = 3
# ======================================================================================================================
# Here we specify the remaining parameters, common to all analyses

# The behavior we use for reference for the control behaviors (those before or after the manipulation, depending
# on the test type).  This will not not affect the stats for the behaviors we are testing for, so this is can
# be set to whatever behavior you like
beh_ref = 'Q'

# Alpha value for forming confidence intervals and testing for significance
alpha = .05

# ======================================================================================================================
# Here we perform the analyses, generating the pdf as we go
# ======================================================================================================================

pdf = FPDF('L') # Landscape orientation

# Set font for producing plots
plt.rc('font', family='arial', weight='normal', size=15)

# Read in raw transitions
raw_trans = read_raw_transitions_from_excel(pathlib.Path(data_folder) / transition_file)

# Recode behavioral annotations
raw_trans = recode_beh(raw_trans, 'beh_before')
raw_trans = recode_beh(raw_trans, 'beh_after')

for cell_type, cell_ids in cell_types:

    # Read in the neural activity for the specified cell type and specify the segment codes to use
    if cell_type == 'a00c':
        a4_act_file = a00c_a4_act_data_file
        a9_act_file = a00c_a9_act_data_file
        seg_codes = A00C_SEG_CODES
    elif cell_type == 'basin':
        a4_act_file = basin_a4_act_data_file
        a9_act_file = basin_a9_act_data_file
        seg_codes = BASIN_SEG_CODES
    elif cell_type == 'handle':
        a4_act_file = handle_a4_act_data_file
        a9_act_file = handle_a9_act_data_file
        seg_codes = HANDLE_SEG_CODES
    else:
        raise (ValueError('The cell type ' + cell_type + ' is not recogonized.'))

    a4_act = scipy.io.loadmat(pathlib.Path(data_folder) / a4_act_file, squeeze_me=True)
    a9_act = scipy.io.loadmat(pathlib.Path(data_folder) / a9_act_file, squeeze_me=True)

    # Correct mistake in labeling if we need to
    if cell_type == 'basin' or cell_type == 'handle':
        ind = np.argwhere(a4_act['newTransitions'] == '0824L2CL')[1][0]
        a4_act['newTransitions'][ind] = '0824L2-2CL'

    for cut_off_time in cut_off_times:
        for manip_type in manip_types:
            print('==========================================================================================')
            print('Producing results for cell type ' + cell_type + ', manipulation_type ' + manip_type +
                  ', cut off time ' + str(cut_off_time) + ', and cell ids: ' + str(cell_ids))

            # Extract transitions
            trans, _ = extract_transitions(raw_trans, cut_off_time)

            # Generate table of data
            a4table = generate_transition_dff_table(act_data=a4_act, trans=trans)
            a9table = generate_transition_dff_table(act_data=a9_act, trans=trans)

            # Put the tables together
            a4table['man_tgt'] = 'A4'
            a9table['man_tgt'] = 'A9'
            data = a4table.append(a9table, ignore_index=True)

            # Down select for manipulation target if needed
            if manip_type == 'A4':
                data = data[data['man_tgt'] == 'A4']
            elif manip_type == 'A9':
                data = data[data['man_tgt'] == 'A9']

            # Down select for cell id
            if isinstance(cell_ids, list):
                keep_rows = data['cell_id'].apply(lambda x: x in set(cell_ids))
                data = data[keep_rows]
                print('Using only cell ids ' + str(cell_ids) + ', leaving ' + str(len(data)) + ' data rows.')
            else:
                print('Using all cell ids, leaving ' + str(len(data)) + ' data rows.')

            # Determine which behaviors are present before and after the manipulation
            trans_subj_cnts = count_unique_subjs_per_transition(data)

            if (test_type == 'state_dependence') or (test_type == 'before_reporting'):
                after_beh_th = 0
                before_beh_th = min_n_subjects_per_beh
            elif ((test_type == 'prediction_dependence') or (test_type == 'after_reporting')
                  or (test_type == 'decision_dependence')):
                after_beh_th = min_n_subjects_per_beh
                before_beh_th = 0
            else:
                raise(ValueError('The test_type ' + test_type + ' is not recognized.'))
            after_beh_sum = trans_subj_cnts.sum()
            after_behs = [b for b in after_beh_sum[after_beh_sum > after_beh_th].index]

            before_beh_sum = trans_subj_cnts.sum(1)
            before_behs = [b for b in before_beh_sum[before_beh_sum > before_beh_th].index]

            # Keep only events that have a transition fron one of the before behaviors into one of the after behaviors
            before_keep_rows = data['beh_before'].apply(lambda x: x in set(before_behs))
            after_keep_rows = data['beh_after'].apply(lambda x: x in set(after_behs))
            data = data[before_keep_rows & after_keep_rows]

            # Update our list of before and after behaviors (since by removing rows, some of our control behaviors
            # may no longer be present
            new_trans_sub_cnts = count_unique_subjs_per_transition(data)
            new_after_beh_sum = new_trans_sub_cnts.sum()
            after_behs = [b for b in new_after_beh_sum[new_after_beh_sum > 0].index]
            new_before_beh_sum = new_trans_sub_cnts.sum(1)
            before_behs = [b for b in new_before_beh_sum[new_before_beh_sum > 0].index]
            print('Using the following before behaviors: ' + str(before_behs))
            print('Using the following after behaviors: ' + str(after_behs))
            print(['Number of rows remaining in data: ' + str(len(data))])

            # Pull out Delta F/F
            if (test_type == 'state_dependence') or (test_type == 'after_reporting'):
                dff = data['dff_after'].to_numpy()
                print('Extracting dff after the manipulation.')
            elif (test_type == 'prediction_dependence') or (test_type == 'before_reporting'):
                dff = data['dff_before'].to_numpy()
                print('Extracting dff before the manipulation.')
            elif test_type == 'decision_dependence':
                dff = data['dff_during'].to_numpy()
                print('Extracting dff during the manipulation.')
            else:
                raise(ValueError('The test_type ' + test_type + ' is not recognized.'))

            # Find grouping of data by subject
            unique_ids = data['subject_id'].unique()
            g = np.zeros(len(data))
            for u_i, u_id in enumerate(unique_ids):
                g[data['subject_id'] == u_id] = u_i

            # Fit models and calculate stats
            if (test_type == 'state_dependence') or (test_type == 'before_reporting'):
                test_behs = before_behs
                control_behs = after_behs
                print('Setting test behaviors to those before the manipulation.')
            elif ((test_type == 'prediction_dependence') or (test_type == 'after_reporting') or
                  (test_type == 'decision_dependence')):
                test_behs = after_behs
                control_behs = before_behs
                print('Setting test behaviors to those after the manipulation.')
            else:
                raise(ValueError('The test_type ' + test_type + ' is not recognized.'))

            control_behs_ref = list(set(control_behs).difference(beh_ref))

            test_behs = sorted(test_behs)
            n_test_behs = len(test_behs)
            test_betas = np.zeros(n_test_behs)
            test_c_ints = np.zeros([2, n_test_behs])
            test_sig = np.zeros(n_test_behs, dtype=np.bool)
            for b_i, b in enumerate(test_behs):
                if (test_type == 'state_dependence') or (test_type == 'before_reporting'):
                    one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(data, beh_before=[b], beh_after=control_behs_ref)
                    pull_ind = 0
                elif ((test_type == 'prediction_dependence') or (test_type == 'after_reporting') or
                      (test_type == 'decision_dependence')):
                    one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(data, beh_before=control_behs_ref, beh_after=[b])
                    pull_ind = len(one_hot_vars_ref)-1
                else:
                    raise(ValueError('The test_type ' + test_type + ' is not recognized.'))

                one_hot_data_ref = np.concatenate([one_hot_data_ref, np.ones([one_hot_data_ref.shape[0], 1])], axis=1)
                one_hot_vars_ref = one_hot_vars_ref + ['ref']

                _, v, _ = np.linalg.svd(one_hot_data_ref)
                if not np.min(v) < .001:

                    beta, acm, n_gprs = grouped_linear_regression_ols_estimator(x=one_hot_data_ref, y=dff, g=g)
                    stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_gprs, alpha=alpha)

                    test_betas[b_i] = beta[pull_ind]
                    test_c_ints[:, b_i] = stats['c_ints'][:, pull_ind]
                    test_sig[b_i] = stats['non_zero'][pull_ind]
                else:
                    test_betas[b_i] = np.nan
                    test_c_ints[:, b_i] = np.nan
                    test_sig[b_i] = False

            # Generate image of fit results
            visualize_coefficient_stats(var_strs=test_behs, theta=test_betas, c_ints=test_c_ints, sig=test_sig,
                                        x_axis_rot=0)
            plt.ylabel('$\Delta F / F$')
            plt.xlabel('Behavior')
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            fig_name_base = cell_type + '_' + manip_type + '_' + str(cut_off_time) + '_' + str(cell_ids)
            fig_path = pathlib.Path(temp_folder) / (fig_name_base + '.png')
            plt.savefig(fig_path)
            plt.close()

            # Add page to the pdf for these results
            if isinstance(cell_ids, list):
                seg_strs = ''
                for vl in cell_ids:
                    seg_strs += seg_codes[vl] + ', '
            else:
                seg_strs = 'all'
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 7, cell_type, ln=1)
            pdf.set_font('Arial',  size=12)
            pdf.cell(40, 5, 'Manipulation target(s): ' + manip_type, ln=1)
            pdf.cell(40, 5, 'Cell ids: ' + seg_strs, ln=1)
            pdf.cell(40, 5, 'Test type: ' + test_type, ln=1)
            pdf.cell(40, 5, 'Cut off time: ' + str(cut_off_time) + ' seconds', ln=1)
            pdf.cell(40, 5, 'Min number of subjects per behavior: ' + str(min_n_subjects_per_beh), ln=1)
            pdf.ln(10)
            pdf.image(str(fig_path), w=170)

#  Output the pdf
pdf.output(pdf_save_path, 'F')




