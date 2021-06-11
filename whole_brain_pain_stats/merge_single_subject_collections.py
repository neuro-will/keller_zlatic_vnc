""" A script for merging collections into one.

This script currently expects that the
        1) Original model fits
        2) Folders with images (which are the individual collections)

will all be in the same folder.  The logic of this basic script looks for all the original model fit results
and forms paths to (2) based on this.

"""

import glob
from pathlib import Path
import re

from keller_zlatic_vnc.collections import merge_collections

# Folder containing the original results and individual collections
rs_folder = r'A:\projects\keller_vnc\results\single_subject_pain_maps'

# Folder where we should save the merged collection into
tgt_folder = r'A:\projects\keller_vnc\results\single_subject_pain_maps\single_subject_pain_maps'

# A new description for the merged collection.
new_description = ('In this analysis we look at results for the single EM specimen, analyzing spontaneous response ' +
               'to stimulus.  We sweep a window in time, to see how encoding changes relative to stimulus onset' +
               'or offsets. The key script to run the statistical tests used to produce these maps is' +
               'calc_single_subject_pain_stats.py and make_single_subj_pain_maps.py was used to generate the' +
               'actual maps.')

# ======================================================================================================================
# Code goes here
# ======================================================================================================================

# Find all results
rs_files = glob.glob(str(Path(rs_folder) / 'pain_maps*.pkl'))

collections = [Path(f).parents[0] / (Path(f).stem + '_images') for f in rs_files]

merge_collections(collections, tgt_folder, new_desc=new_description, ignore_extensions=['.png'])