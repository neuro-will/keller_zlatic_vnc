""" Fits models with only interaction terms, producing a pdf with all the possible ways we can make these fits. """

import copy
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
pdf_save_path = '/Users/williambishop/Desktop/trans_dep.pdf'


# ======================================================================================================================
# Here we specify a path to a folder we can use for saving temporary files (images)
temp_folder = '/Users/williambishop/Desktop/trans_dep'

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
periods = ['after', 'before']
manip_types = ['A4', 'A9', 'A4+A9']
cut_off_times = [3.656, 9.0034]

# ======================================================================================================================
# Here we specify the remaining parameters, common to all analyses

# Define how many subjects we need to observe a transition from to include in the model
min_n_trans_subjs = 3

# Colors to assoicate with behaviors
clrs = {'F': np.asarray([255, 128, 0])/255,
        'B': np.asarray([0, 0, 153])/255,
        'Q': np.asarray([255, 51, 153])/255,
        'O': np.asarray([204, 153, 255])/255,
        'T': np.asarray([0, 204, 0])/255,
        'P': np.asarray([0, 153, 153])/255,
        'H': np.asarray([52, 225, 235])/255}

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
        for period in periods:
            for manip_type in manip_types:
                print('==========================================================================================')
                print('Producing results for cell type ' + cell_type + ', period ' + period +
                      ', manipulation_type ' + manip_type + ', and cut off time ' + str(cut_off_time))

                # Extract transitions
                trans = extract_transitions(raw_trans, cut_off_time)

                # Generate table of data
                a4table = generate_transition_dff_table(act_data=a4_act, trans=trans)
                a9table = generate_transition_dff_table(act_data=a9_act, trans=trans)

                # Put the tables together
                a4table['man_tgt'] = 'A4'
                a9table['man_tgt'] = 'A9'
                data = a4table.append(a9table, ignore_index=True)

                print('Length of data before further processing: ' + str(len(data)))

                # Down select for manipulation target if needed
                if manip_type == 'A4':
                    data = data[data['man_tgt'] == 'A4']
                elif manip_type == 'A9':
                    data = data[data['man_tgt'] == 'A9']

                # Count the number of subjects displaying each behavioral transition
                trans_subj_cnts = count_unique_subjs_per_transition(data)

                # Get list of transitions with the minimum number of subjects
                mdl_trans = []
                for from_beh in trans_subj_cnts.index:
                    for to_beh in trans_subj_cnts.columns:
                        if trans_subj_cnts[to_beh][from_beh] >= min_n_trans_subjs:
                            mdl_trans.append((from_beh, to_beh))

                # Remove any events that do not display one of the transitions we include in the model
                l_data = len(data)
                keep_rows = np.zeros(l_data, dtype=np.bool)
                for r_i, r_index in enumerate(data.index):
                    row_trans = (data['beh_before'][r_index], data['beh_after'][r_index])
                    if row_trans in mdl_trans:
                        keep_rows[r_i] = True

                data = data[keep_rows]

                print('Length of data after processing ' + str(len(data)))

                # Pull out Delta F/F
                if period == 'before':
                    dff = data['dff_before'].to_numpy()
                elif period == 'after':
                    dff = data['dff_after'].to_numpy()
                else:
                    raise (ValueError('The period ' + ' period is not recogonized.'))

                # Find grouping of data by subject
                unique_ids = data['subject_id'].unique()
                g = np.zeros(len(data))
                for u_i, u_id in enumerate(unique_ids):
                    g[data['subject_id'] == u_id] = u_i

                # Perform fitting and statistics
                one_hot_data, one_hot_vars = one_hot_from_table(data, beh_before=[], beh_after=[],
                                                                enc_subjects=False, enc_beh_interactions=False,
                                                                beh_interactions=mdl_trans)

                beta, acm, n_gprs = grouped_linear_regression_ols_estimator(x=one_hot_data, y=dff, g=g)
                stats = grouped_linear_regression_acm_stats(beta=beta, acm=acm, n_grps=n_gprs, alpha=alpha)

                # Generate image for coefficents grouped by behavior before manipulation
                before_order, before_clrs = order_and_color_interaction_terms(terms=[t[-2:] for t in one_hot_vars],
                                                                              colors=clrs, sort_by_before=True)
                visualize_coefficient_stats(var_strs=[one_hot_vars[i] for i in before_order],
                                            theta=beta[before_order], c_ints=stats['c_ints'][:, before_order],
                                            sig=stats['non_zero'][before_order],
                                            var_clrs=before_clrs)

                fig_name_base = cell_type + '_' + period + '_' + manip_type + '_' + str(cut_off_time)
                plt.tight_layout()
                plt.ylabel('$\Delta F / F$')
                plt.title('Grouped by Preceeding Behavior')
                fig = plt.gcf()
                fig.set_size_inches(8, 6)
                pb_path = pathlib.Path(temp_folder) / (fig_name_base + '_pb.png')
                plt.savefig(pb_path)
                plt.close()

                # Generate image for coefficents grouped by behavior after manipulation
                after_order, after_clrs = order_and_color_interaction_terms(terms=[t[-2:] for t in one_hot_vars],
                                                                            colors=clrs, sort_by_before=False)
                visualize_coefficient_stats(var_strs=[one_hot_vars[i] for i in after_order],
                                            theta=beta[after_order], c_ints=stats['c_ints'][:, after_order],
                                            sig=stats['non_zero'][after_order],
                                            var_clrs=after_clrs)
                plt.tight_layout()
                plt.ylabel('$\Delta F / F$')
                plt.title('Grouped by Succeeding Behavior')
                fig = plt.gcf()
                fig.set_size_inches(8, 6)
                sb_path = pathlib.Path(temp_folder) / (fig_name_base + '_sb.png')
                plt.savefig(sb_path)
                plt.close()

                # Add page to the pdf for these results
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 7, cell_type, ln=1)
                pdf.set_font('Arial',  size=12)
                pdf.cell(40, 5, 'DFF ' + period + ' manipulation', ln=1)
                pdf.cell(40, 5, 'Manipulation target(s): ' + manip_type, ln=1)
                pdf.cell(40, 5, 'Cut off time: ' + str(cut_off_time) + ' seconds', ln=1)
                pdf.ln(20)
                image_x = pdf.get_x()
                image_y = pdf.get_y()
                pdf.image(str(pb_path), x=image_x, y=image_y, w=130) # 180
                pdf.image(str(sb_path), x=image_x + 140, y=image_y, w=130)

# Output the pdf
pdf.output(pdf_save_path, 'F')