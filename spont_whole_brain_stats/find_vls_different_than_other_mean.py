"""
A script for batch post-processing a set of initial model fits.

"""

import copy
import glob
from pathlib import Path
import re

from keller_zlatic_vnc.whole_brain.spontaneous import parallel_test_for_diff_than_mean_vls

# ======================================================================================================================
# Base parameters go here.
#
# These will be modified and added to when passing them to the post-processing function for each result
# ======================================================================================================================
# Basic parameters go here
base_ps = dict()

# Provide a folder for which we should search for results to post process
base_ps['save_folder'] = r'A:\projects\keller_vnc\results\single_subject\spont_window_sweep_v2\ind_collections'

# ======================================================================================================================
# Code starts here
# ======================================================================================================================

# Search for files to post-process
pkl_files = glob.glob(str(Path(base_ps['save_folder']) / '*.pkl'))

# Filter out post processed results
pkl_files = [f for f in pkl_files if re.match('.*mean_cmp_stats.pkl', f) is None]

n_results = len(pkl_files)

if __name__ == '__main__':
    for f_i, f in enumerate(pkl_files):
        ps = copy.deepcopy(base_ps)
        ps['basic_rs_file'] = Path(f).name

        print('****************************************************************************************')
        print('Post-processing results ' + str(f_i + 1) + ' of ' + str(n_results) + '.')
        parallel_test_for_diff_than_mean_vls(ps=ps)



