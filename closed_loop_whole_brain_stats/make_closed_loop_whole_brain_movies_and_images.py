""" Generates images and movies of maps.

This script assumes:

1) Initial model fitting was perfomed by the script fit_initial_models.py

2) Post-processing was performed by the script find_vls_different_than_other_mean.py

"""

import glob
import os
from pathlib import Path
import pickle
import re
import numpy as np

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs

# Specify folder results are saved in
results_folder = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\results\new_whole_brain_stats\5_25_25\decision_dependence_5_25_25_no_clustered_errors'

# Switch for setting additional parameters below based on if we are making images for the initial stats or after
# comparing each coefficent to the mean of the others in its group
stat_types = 'initial' # 'initial' or 'mean_cmp'

multi_cmp_type = 'by' #'none', 'bon' or 'by'

# Provide a string suffix specifying the results file
if stat_types == 'initial':
    rs_str = '.*\d+.pkl'
else:
    rs_str = '.*_mean_cmp_stats.pkl'

# Specify location of overlay files - these are for max projections
overlay_files = [r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\overlays\horz_mean.png',
                 r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\overlays\cor_mean.png',
                 r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\data\overlays\sag_mean.png']

# The string p-values are stored under: 'non_zero_p' or 'eq_mean_p'
if stat_types == 'initial':
    p_vl_str = 'non_zero_p'
else:
    p_vl_str = 'eq_mean_p'

if multi_cmp_type == 'bon':
    p_vl_str += '_corrected_bon'
elif multi_cmp_type == 'by':
    p_vl_str += '_corrected_by'

# Lower percentage of p-values that brightness saturates at - should be between 0 and 100
min_p_vl_perc = .01

# Name of the roi group the results were generated for - currently, we can only generate results in batch if
# they are all for the same rois
roi_group = 'rois_5_25_25'

ex_dataset_file = r'W:\\SV4\\CW_17-08-23\\L1-561nm-ROIMonitoring_20170823_145226.corrected\\extracted\\dataset.pkl'

# Find results to generate images and maps for
results_files = glob.glob(str(Path(results_folder) / '*.pkl'))
results_files = [f for f in results_files if re.match(rs_str, f)]

# Filter our results we already have images for
results_files = [f for f in results_files if not os.path.exists(Path(f).parent / (Path(f).stem + '_images'))]
n_results_files = len(results_files)
print('Generating images and videos for ' + str(n_results_files) + ' results.')

for f in results_files:

    # Determine where we will save the results
    save_folder_path = Path(f).parent / (Path(f).stem + '_images')

    # Load the results
    with open(f, 'rb') as f:
        rs = pickle.load(f)

    # Save the number of transitions
    os.makedirs(save_folder_path)
    rs['n_trans'].to_csv(save_folder_path / 'n_trans.csv')
    rs['n_subjs_per_trans'].to_csv(save_folder_path / 'n_subjs_per_trans.csv')

    # Put the results in the format expected for the plotting function
    n_rois = len(rs['full_stats'])
    p_vls = np.stack([s[p_vl_str] for s in rs['full_stats']])
    beta = np.stack([s['beta'] for s in rs['full_stats']])

    beh_stats = {b: {'p_values': p_vls[:, b_i], 'beta': beta[:, b_i]} for b_i, b in enumerate(rs['var_names'])}
    plot_rs = {'beh_stats': beh_stats}

    # Generate images and movies
    make_whole_brain_videos_and_max_projs(rs=plot_rs,
                                          save_folder_path=save_folder_path,
                                          overlay_files=overlay_files,
                                          roi_group=roi_group,
                                          save_supp_str='',
                                          gen_mean_movie=False,
                                          gen_mean_tiff=False,
                                          gen_coef_movies=False,
                                          gen_coef_tiffs=True,
                                          gen_p_value_movies=False,
                                          gen_p_value_tiffs=True,
                                          gen_filtered_coef_movies=False,
                                          gen_filtered_coef_tiffs=False,
                                          gen_combined_movies=False,
                                          gen_combined_tiffs=True,
                                          gen_combined_projs=False,
                                          gen_uber_movies=True,
                                          min_p_val_perc=min_p_vl_perc,
                                          max_p_vl=.05,
                                          ex_dataset_file=ex_dataset_file)

