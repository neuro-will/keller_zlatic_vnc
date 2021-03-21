""" Script to make movies and images for whole-brain spontaneous behavior statistics.

This script assumes the basic statistical calculations have been performed
with spont_events_initial_stats_calculation.ipynb.  It then generates whole-brain movies and imaages
showing the results.

"""

from pathlib import Path
import pickle

import numpy as np

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs


# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

# File with calculated statistics
results_file = r'\\dm11\bishoplab\projects\keller_vnc\results\single_subject_spont_stats\spont_1_5_5_long_bl_co_4_start_locked_0_6_ece_prepooled_post_tc_mean_cmp_stats.pkl'

# Specify location of overlay files
overlay_files = [r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\horz_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\cor_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\sag_mean.png']

# The group of rois the statistics were generated for
roi_group = 'rois_1_5_5'

# The string p-values are stored under
p_vl_str = 'eq_mean_p' # 'non_zero_p' or 'non_max_p' or 'eq_mean_p'

# Lower percentage of p-values that brightness saturates at - should be between 0 and 100
min_p_vl_perc = .0001

#ex_dataset_file=r'K:/SV4/CW_17-08-23/L1-561nm-ROIMonitoring_20170823_145226.corrected/extracted/dataset.pkl'
ex_dataset_file=r'K:\SV4\CW_18-02-15\L1-561nm-openLoop_20180215_163233.corrected\extracted\dataset.pkl'

# ======================================================================================================================
# Code starts here
# ======================================================================================================================

# Determine where we will save the results
save_folder_path = Path(results_file).parent / (Path(results_file).stem + '_images')

# Load the results
with open(results_file, 'rb') as f:
    rs = pickle.load(f)

# Put the results in the format expected for the plotting function
beh_trans = [b[0] + '_' + b[1] for b in rs['beh_trans']]

n_rois = len(rs['full_stats'])
p_vls = np.stack([s[p_vl_str] for s in rs['full_stats']])
beta = np.stack([s['beta'] for s in rs['full_stats']])


beh_stats = {b: {'p_values': p_vls[:, b_i], 'beta': beta[:, b_i]} for b_i, b in enumerate(beh_trans)}
plot_rs = {'beh_stats': beh_stats}

make_whole_brain_videos_and_max_projs(rs=plot_rs,
                                      save_folder_path=save_folder_path,
                                      overlay_files=overlay_files,
                                      roi_group=roi_group,
                                      save_supp_str='',
                                      gen_mean_movie=True,
                                      gen_mean_tiff=False,
                                      gen_coef_movies=False,
                                      gen_coef_tiffs=True,
                                      gen_p_value_movies=False,
                                      gen_p_value_tiffs=True,
                                      gen_filtered_coef_movies=False,
                                      gen_filtered_coef_tiffs=False,
                                      gen_combined_movies=False,
                                      gen_combined_tiffs=True,
                                      gen_combined_projs=True,
                                      gen_uber_movies=True,
                                      min_p_val_perc=min_p_vl_perc,
                                      max_p_vl=.05,
                                      ex_dataset_file=ex_dataset_file)

