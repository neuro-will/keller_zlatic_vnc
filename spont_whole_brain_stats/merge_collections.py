""" A script for merging collections into one.

This script should be run after 'generate_collections.py'

The logic for this script is simple.  The user supplies a base folder where individual collections have already
been created.  This script assumes that in this same base folder .pkl files exist with the fit results (for
both the original model fits as well as mean comparison results) that each collection is based off of.  It will
then look for a folder with images (which in fact now are collections because we assume generate_collections has
already been run) for each .pkl file and merge the collections for all results into one large collection.

The user can specify a new save location where the merged collection should be stored.

"""

import glob
from pathlib import Path
import re

from keller_zlatic_vnc.collections import merge_collections

# Folder containing the original results and individual collections
rs_folder = r'A:\projects\keller_vnc\results\single_subject\new_model_maps_v1\whole_specimen_rois'

# Folder where we should save the merged collection into
tgt_folder = r'A:\projects\keller_vnc\results\single_subject\new_model_maps_v1\whole_specimen_rois_collection'

# A new description for the merged collection.
new_description = ('Results for updated models, which look at how neural encoding depends on behavior both before and ' +
                   'after stimulus.  The maps contained here are for 5x5x1 rois throughout the whole speciment. ' +
                   'The key scripts to run the statistical tests used to produce these ' +
                   'maps are fit_init_models.py and find_vls_different_than_other_mean.py. The script ' +
                   'make_spont_whole_brain_movies_and_images.py was then used to render the actual maps.')

# Specify the type of maps the collections we are merging were made for

# ======================================================================================================================
# Code goes here
# ======================================================================================================================

# Find all results
rs_files = glob.glob(str(Path(rs_folder) / '*.pkl'))

collections = [Path(f).parents[0] / (Path(f).stem + '_images') for f in rs_files]

merge_collections(collections, tgt_folder, new_desc=new_description, ignore_extensions=['.png'])

