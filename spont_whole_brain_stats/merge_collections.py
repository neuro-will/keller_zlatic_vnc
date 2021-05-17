""" A script for merging collections into one.

This script currently expects that the
        1) Original model fits
        2) Post-processed (mean comparisons) results
        3) Folders with images (which are the individual collections)

will all be in the same folder.  The logic of this basic script looks for all the original model fit results
and forms paths to (3) based on this.

"""

import glob
from pathlib import Path
import re

from keller_zlatic_vnc.collections import merge_collections

# Folder containing the original results and individual collections
rs_folder = r'\\dm11\bishoplab\projects\keller_vnc\results\single_subject\spont_window_sweep\ind_collections'

# Folder where we should save the merged collection into
tgt_folder = r'\\dm11\bishoplab\projects\keller_vnc\results\single_subject\spont_window_sweep\spont_window_sweep'

# A new description for the merged collection.
new_description = ('In this analysis we look at results for the single EM specimen, analyzing spontaneous behavior ' +
               'transitions.  We sweep a window in time, to see how encoding changes relative to behavior onset. ' +
               ' We also run analyses with succeeding turns pooled and not pooled.  The plan is to add additional '
               ' variations in the future. ' +
               'The key scripts to run the statistical tests used to produce these maps are fit_init_models.py ' +
               ' and find_vls_different_than_other_mean.py. The script ' +
               'make_spont_whole_brain_movies_and_images.py was then used to render the actual maps.')

# ======================================================================================================================
# Code goes here
# ======================================================================================================================

# Find all results
rs_files = glob.glob(str(Path(rs_folder) / '*.pkl'))
rs_files = [f for f in rs_files if re.match('.*mean_cmp_stats.pkl', f) is None]

collections = [Path(f).parents[0] / (Path(f).stem + '_mean_cmp_stats_images') for f in rs_files]

merge_collections(collections, tgt_folder, new_desc=new_description, ignore_extensions=['.png'])

