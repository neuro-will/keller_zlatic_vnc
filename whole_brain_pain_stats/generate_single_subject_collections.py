""" A script to generate collections after images and movies of maps have been generated.

This script should be called after make_single_subj_pain_maps.

Note a couple of "odd" things:

1) The user needs to supply a path to a file with the parameters that were used for fluorescence extraction.  In
reality, these parameters were saved with each specimen we extracted fluorescence from, so there are many files that
could be provided.  The user has to only supply one.  Note that these will be for a single setting of voxel sizes.

2) The user also has to supply a path to a file with the parameters used to extract baselines.  The same caveats and
guidance in (1) applies here.  Also, in some cases, if baselines were only ever calculated with a single set of
parameters, the relevant parameters may be in the same parameter file with the fluorescence extraction parameters,
so the files provided in (1) and (2) may be the same.

3) This script currently expects that the
        1) Original model fits
        2) Folders with images
    will all be in the same folder.  The logic of this basic script looks for all the original model fit results
    and forms paths to (2) based on this.

"""

import glob
from pathlib import Path
import pickle
import re

from keller_zlatic_vnc.collections import form_collection

# Folder holding results
rs_folder = r'A:\projects\keller_vnc\results\single_subject_pain_maps_v2'

# Give path to a file holding parameters used to extract fluorescece
f_extraction_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\extraction_params.pkl'

# Give path to a file holding parameters used to extract baselines
baseline_calc_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\long_baseline_extract_params.pkl'

# List those who can be contacted with questions about the collection
responsible = ['William Bishop <bishopw@hhmi.janelia.org>',
               'Andrew Champion <andrew.champion@gmail.com>']

# Provide a description of the collection.
description = ('This is the second iteration of looking at the single EM specimen response to pain.' +
               ' Here we still sweep a window in time, to see how encoding changes relative to stimulus onset' +
               ' and offset, while varying the window length.  But now in addition, we also analyze' +
               ' the response to long and short stimulus events separately when using windows aligned to stim offset. ' +
               ' The key script to run the statistical tests used to ' +
               'produce these maps is calc_single_subject_pain_stats.py and make_single_subj_pain_maps.py was ' +
               ' used to generate the actual maps.')

# List hashes identify commits in git for the different pieces of code used to produce these results
git_hashes = {'janelia_core': '0cc52fb406b985b274d222ee16b05ba20365715d',
              'keller_zlatic_vnc': '5320ebcde2c3f7f5ab8b7342b2d3bf0357342d2c'}

# List the parameters that should be included in the metadata file, with comments that should also be included
f_extraction_yaml_fields = {'voxel_size_per_dim': 'Number of voxels in each dimension of a supervoxel.'}

baseline_calc_yaml_fields = {'window_length': 'Length of window used for baseline calculation.',
                             'filter_start': 'Initial offset, relative to first data point, of window used for baseline calculations.',
                             'write_offset': "Offset between first point in window and the point the filtered output is assigned to.",
                             'p': 'The particular percentile used for percentile filtering.'}

mdl_fitting_yaml_fields = {'n_before_tm_pts': 'Length of window in imaging time points for calculating DFF in before the stimulus.',
                           'after_aligned': 'Specifies if the window for calculating DFF after the stimulus was aligned to stimulus onset of offset.',
                           'after_offset': 'The offset in imaging time points between the alignment point and the start of the window for calculating DFF in after the stimulus.',
                           'n_after_tm_pts': 'Length of window in imaging time points for calculating DFF in after the stimulus.'}

# ======================================================================================================================
# Code goes here
# ======================================================================================================================

# Load pickle files containing fluorescence extraction and baseline calculation parameters
with open(f_extraction_params_file, 'rb') as f:
    f_extraction_params = pickle.load(f)

with open(baseline_calc_params_file, 'rb') as f:
    baseline_calc_params = pickle.load(f)

# Find all results
rs_files = glob.glob(str(Path(rs_folder) / 'pain*.pkl'))
n_results = len(rs_files)

for f_i, f in enumerate(rs_files):
    mdl_fitting_params_file = f
    image_folder = Path(f).parents[0] / (Path(f).stem + '_images')

    with open(mdl_fitting_params_file, 'rb') as f:
        mdl_fitting_results = pickle.load(f)


    # Determine the set of preceeding and succeeding behaviors
    preceding_behs = ['G']
    preceding_behs.sort()
    suceeding_behs = ['G']
    suceeding_behs.sort()

    # Form the collection
    params = [{'desc': 'f_extraction_params',
          'for_saving': f_extraction_params,
          'for_metadata': f_extraction_params['roi_extract_opts'],
          'inc_params': f_extraction_yaml_fields},
          {'desc': 'baseline_calc_params',
          'for_saving': baseline_calc_params,
          'for_metadata': baseline_calc_params['baseline_calc_opts'],
          'inc_params': baseline_calc_yaml_fields},
          {'desc': 'mdl_fitting_params',
           'for_saving': mdl_fitting_results['ps'],
           'for_metadata': mdl_fitting_results['ps'],
           'inc_params': mdl_fitting_yaml_fields}]

    form_collection(image_folder=image_folder,
                    tgt_folder=image_folder,
                    description=description,
                    responsible=responsible,
                    git_hashes=git_hashes,
                    preceding_behs=preceding_behs,
                    suceeding_behs=suceeding_behs,
                    params=params,
                    ignore_extensions=['.png'])

    print('Done forming collection ' + str(f_i+1) + ' of ' + str(n_results) + '.')




