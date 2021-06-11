""" Generates images and movies of maps.

This script assumes:

1) Initial model fitting was perfomed by the script calc_single_subject_pain_stats.py

"""

import glob
from pathlib import Path
import pickle

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs

# Specify folder results are saved in

results_folder = r'A:\projects\keller_vnc\results\single_subject_pain_maps'

# Provide a string for specifying the results file
rs_reg_str = 'pain_maps*.pkl'

# Name of the roi group the results were generated for - currently, we can only generate results in batch if
# they are all for the same rois
roi_group = 'rois_1_5_5'

ex_dataset_file = r'K:\SV4\CW_18-02-15\L1-561nm-openLoop_20180215_163233.corrected\extracted\dataset.pkl'

# P-value thresholds for producing thresholded coefficient maps
p_vl_thresholds = [.05, .95]

# Limits when visualizing coefficients
coef_lims = [-1.5, 1.5]

# Find results to generate images and maps for
results_files = glob.glob(str(Path(results_folder) / rs_reg_str))

for f in results_files:

    # Determine where we will save the results
    save_folder_path = Path(f).parent / (Path(f).stem + '_images')

    # Load the results
    with open(f, 'rb') as f:
        rs = pickle.load(f)

    # Generate images and movies
    make_whole_brain_videos_and_max_projs(rs=rs['full_stats'],
                                          save_folder_path=save_folder_path,
                                          overlay_files=None,
                                          roi_group=roi_group,
                                          save_supp_str='',
                                          gen_mean_movie=True,
                                          gen_mean_tiff=True,
                                          gen_coef_movies=False,
                                          gen_coef_tiffs=True,
                                          gen_p_value_movies=False,
                                          gen_p_value_tiffs=True,
                                          gen_filtered_coef_movies=False,
                                          gen_filtered_coef_tiffs=True,
                                          gen_combined_movies=False,
                                          gen_combined_tiffs=True,
                                          p_vl_thresholds = p_vl_thresholds,
                                          gen_combined_projs=False,
                                          gen_uber_movies=True,
                                          max_p_vl=.05,
                                          coef_lims=coef_lims,
                                          ex_dataset_file=ex_dataset_file)
