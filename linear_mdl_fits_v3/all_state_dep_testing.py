""" Tests for state dependence in DFF, looking across multiple different ways of processing data. """

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
from keller_zlatic_vnc.linear_modeling import one_hot_from_table
from keller_zlatic_vnc.linear_modeling import order_and_color_interaction_terms
from keller_zlatic_vnc.linear_modeling import reference_one_hot_to_beh


from janelia_core.stats.regression import grouped_linear_regression_ols_estimator
from janelia_core.stats.regression import grouped_linear_regression_acm_stats
from janelia_core.stats.regression import visualize_coefficient_stats

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

ps = dict()

# ======================================================================================================================
# Here we specify where the pdf should be saved and its name in a single path
pdf_save_path = '/Users/williambishop/Desktop/test_1.pdf'


# ======================================================================================================================
# Here we specify a path to a folder we can use for saving temporary files (images)
temp_folder = '/Users/williambishop/Desktop/temp_1'

# ======================================================================================================================
# Here we specify the location of the data

data_folder = r'/Users/williambishop/Desktop/extracted_dff_v2'
transition_file = 'transition_list.xlsx'

a00c_a4_act_data_file = 'A00c_activity_A4.mat'
a00c_a9_act_data_file = 'A00c_activity_A9.mat'

basin_a4_act_data_file = 'Basin_activity_A4.mat'
basin_a9_act_data_file = 'Basin_activity_A9.mat'

handle_a4_act_data_file = 'Handle_activity_A4.mat'
handle_a9_act_data_file = 'Handle_activity_A9.mat'

# ======================================================================================================================
# Here, we specify the different ways we want to filter the data when fitting models.  We will produce results for all
# possible combinations of the options below.

cell_types = ['a00c', 'handle', 'basin']
manip_types = ['A4', 'A9', 'A4+A9']
cut_off_times = [3.656, 9.0034]

# ======================================================================================================================
# Here we specify the remaining parameters, common to all analyses

# The behavior after the manipulation we use for reference - this will not affect the calculate stats for the
# behaviors before the manipulation, which is what we care about, so this is arbitrary
after_beh_ref = 'Q'

# Alpha value for forming confidence intervals and testing for significance
alpha = .05

# ======================================================================================================================
# Here we perform the analyses, generating the pdf as we go
# ======================================================================================================================

pdf = FPDF('L')

# Set font for producing plots
plt.rc('font', family='arial', weight='normal', size=15)

# Read in raw transitions
raw_trans = read_raw_transitions_from_excel(pathlib.Path(data_folder) / transition_file)

# Recode behavioral annotations
raw_trans = recode_beh(raw_trans, 'Beh Before')
raw_trans = recode_beh(raw_trans, 'Beh After')

for cell_type in cell_types:

    # Read in the neural activity for the specified cell type
    if cell_type == 'a00c':
        a4_act_file = a00c_a4_act_data_file
        a9_act_file = a00c_a9_act_data_file
    elif cell_type == 'basin':
        a4_act_file = basin_a4_act_data_file
        a9_act_file = basin_a9_act_data_file
    elif cell_type == 'handle':
        a4_act_file = handle_a4_act_data_file
        a9_act_file = handle_a9_act_data_file
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
                  ', and cut off time ' + str(cut_off_time))

            # Extract transitions
            trans = extract_transitions(raw_trans, cut_off_time)

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

            # Determine which behaviors are present before and after the manipulation
            trans_subj_cnts = count_unique_subjs_per_transition(data)

            after_beh_sum = trans_subj_cnts.sum()
            after_behs = [b for b in after_beh_sum[after_beh_sum > 0].index]

            before_beh_sum = trans_subj_cnts.sum(1)
            before_behs = [b for b in before_beh_sum[before_beh_sum > 0].index]

            # Pull out Delta F/F
            dff = data['dff_after'].to_numpy()

            # Find grouping of data by subject
            unique_ids = data['subject_id'].unique()
            g = np.zeros(len(data))
            for u_i, u_id in enumerate(unique_ids):
                g[data['subject_id'] == u_id] = u_i

            # Fit models and calculate stats
            after_behs_ref = list(set(after_behs).difference(after_beh_ref))

            n_before_behs = len(before_behs)
            before_betas = np.zeros(n_before_behs)
            before_c_ints = np.zeros([2, n_before_behs])
            before_sig = np.zeros(n_before_behs, dtype=np.bool)
            for b_i, b in enumerate(before_behs):
                one_hot_data_ref, one_hot_vars_ref = one_hot_from_table(data, beh_before=[b], beh_after=after_behs_ref)
                one_hot_data_ref = np.concatenate([one_hot_data_ref, np.ones([one_hot_data_ref.shape[0], 1])], axis=1)
                one_hot_vars_ref = one_hot_vars_ref + ['ref']

                _, v, _ = np.linalg.svd(one_hot_data_ref)
                if np.min(v) < .001:
                    raise (RuntimeError('regressors are nearly co-linear'))

                beta, acm, n_gprs = grouped_linear_regression_ols_estimator(x=one_hot_data_ref, y=dff, g=g)
                stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_gprs, alpha=.05)

                before_betas[b_i] = beta[0]
                before_c_ints[:, b_i] = stats['c_ints'][:, 0]
                before_sig[b_i] = stats['non_zero'][0]

            # Generate image of fit results
            visualize_coefficient_stats(var_strs=before_behs, theta=before_betas, c_ints=before_c_ints, sig=before_sig,
                                        x_axis_rot=0)
            plt.ylabel('$\Delta F / F$')
            plt.xlabel('Behavior Before Manipulation')
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            fig_name_base = cell_type + '_' + manip_type + '_' + str(cut_off_time)
            fig_path = pathlib.Path(temp_folder) / (fig_name_base + '.png')
            plt.savefig(fig_path)
            plt.close()

            # Add page to the pdf for these results
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 7, cell_type, ln=1)
            pdf.set_font('Arial',  size=12)
            pdf.cell(40, 5, 'Manipulation target(s): ' + manip_type, ln=1)
            pdf.cell(40, 5, 'Cut off time: ' + str(cut_off_time) + ' seconds', ln=1)
            pdf.ln(10)
            pdf.image(str(fig_path), w=200)

#  Output the pdf
pdf.output(pdf_save_path, 'F')




