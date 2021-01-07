# Python script for generating additional movies and images after original processing

import glob
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re

import numpy as np

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs

# ======================================================================================================================
# List directories holding results - we will produce images for all results in these
# directories matching the rest of the criteria the user has specified below
base_dirs = [r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_1_5_5_long_bl']

# List which type of tests we want to produce images for
test_types = ['state_dependence', 'prediction_dependence', 'decision_dependence']

# Specify where we find overlay files
overlay_files = [r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\horz_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\cor_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\sag_mean.png']

# Max number of processes that can run at once
max_n_cpu = 10

# ======================================================================================================================
# Get a list of all matching files in each directory

re_pattern = '(' + '|'.join(test_types) + ').*'

matching_files = []
for b_dir in base_dirs:
    all_files = glob.glob(str(Path(b_dir) / '*.pkl'))
    matching_files.append([f for f in all_files if re.match(re_pattern, Path(f).name) is not None])

matching_files = list(itertools.chain(*matching_files))
n_matching_files = len(matching_files)

# ======================================================================================================================
# Define helper functions

def gen_images(results_file):

    # See if we have already completed results for this file
    status_file_name = '.mo_status_' + Path(results_file).name
    status_file_path = Path(results_file).parents[0] / status_file_name

    #if os.path.exists(status_file_path):
    #    print('Skip')
    #else:
    with open(results_file, 'rb') as f:
        rs = pickle.load(f)
    ps = rs['ps']

    data_file_stem = Path(ps['data_file']).stem
    save_str = ps['save_str'] + '_' + data_file_stem


    roi_group = 'rois_' + re.match('.*dff_([0-9]{1,2}_[0-9]{1,2}_[0-9]{1,2}).*', Path(results_file).name).group(1)

    make_whole_brain_videos_and_max_projs(results_file=Path(results_file),
                                              overlay_files=overlay_files,
                                              save_supp_str=save_str,
                                              roi_group=roi_group,
                                              gen_mean_tiff=False, gen_mean_movie=False,
                                              gen_coef_movies=False, gen_coef_tiffs=False,
                                              gen_p_value_movies=False, gen_p_value_tiffs=False,
                                              gen_filtered_coef_movies=False, gen_filtered_coef_tiffs=False,
                                              gen_combined_tiffs=True, gen_combined_movies=False,
                                              gen_combined_projs=False, gen_uber_movies=False)

    # Save a small marker file noting that we are done
    with open(status_file_path, 'wb') as f:
        pickle.dump({'done': True}, f)

# ======================================================================================================================
# Run everything in parallel


if __name__ == '__main__':

    # ==============================================================================
    # See how many processors we have available
    cpu_count = mp.cpu_count()
    n_used_cpus = np.min([cpu_count, n_matching_files])
    if n_used_cpus > max_n_cpu:
        n_used_cpus = max_n_cpu
    print('Processing ' + str(n_matching_files) + ' files with ' + str(n_used_cpus) + ' processes.')

    pool = mp.Pool(n_used_cpus, maxtasksperchild=1)
    pool.map(gen_images, matching_files, chunksize=1)
