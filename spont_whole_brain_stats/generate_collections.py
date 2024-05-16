""" A script to generate collections after images and movies of maps have been generated.

This script should be called after make_spont_whole_brain_movies_and_images has been called to make maps both
for the original model fits and for the mean comparison results.

This script will:

1) Search through a base folder for all .pkl files (we assume these pkl files hold statistical results for either
the original model fits or the mean comparison post processing of a model fit).

2) Assume that maps have been rendered for each .pkl file found in (1) and that these are stored in a folder with
the same name as the .pkl file

3) Create a collection in each folder in (2)

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
        2) Post-processed (mean comparisons) results
        3) Folders with images
    will all be in the same folder.  The logic of this basic script looks for all the original model fit results
    and forms paths to (2) and (3) based on this.

"""

import glob
from pathlib import Path
import pickle
import re

from keller_zlatic_vnc.collections import form_collection

# Folder holding results
rs_folder = r'Z:\Exchange\Will\bishoplab\projects\keller_drive\keller_vnc\results\single_subject\new_bl\brain_only'

# Give path to a file holding parameters used to extract fluorescece
f_extraction_params_file = r'W:\SV4\CW_18-02-15\L1-561nm-openLoop_20180215_163233.corrected\extracted\brain_rois_1_5_5\extraction_params.pkl'

# Give path to a file holding parameters used to extract baselines
baseline_calc_params_file = r'W:\SV4\CW_18-02-15\L1-561nm-openLoop_20180215_163233.corrected\extracted\brain_rois_1_5_5\baseline_161_extract_params.pkl'

# List those who can be contacted with questions about the collection
responsible = ['William Bishop <bishopw@hhmi.janelia.org>',
               'Andrew Champion <andrew.champion@gmail.com>']

# Provide a description of the collection.
description = ('Results for updated models, which look at how neural encoding depends on behavior both before and ' +
               'after stimulus.  Here we use a shorter baseline 161 than before and focus only on the brain.')

# List hashes identify commits in git for the different pieces of code used to produce these results
git_hashes = {'janelia_core': 'ac16ae27170fb304d65d8ab72cf583efc51a3513',
              'keller_zlatic_vnc': '96e8059f6206830993a897ad78e62b637a0d4e7d'}

# List the parameters that should be included in the metadata file, with comments that should also be included
f_extraction_yaml_fields = {'voxel_size_per_dim': 'Number of voxels in each dimension of a supervoxel.'} # 'segmentation_file': 'File segmentations were saved in.'

baseline_calc_yaml_fields = {'window_length': 'Length of window used for baseline calculation.',
                             'filter_start': 'Initial offset, relative to first data point, of window used for baseline calculations.',
                             'write_offset': "Offset between first point in window and the point the filtered output is assigned to.",
                             'p': 'The particular percentile used for percentile filtering.'}

mdl_fitting_yaml_fields = {
    'clean_event_def': 'The criteria used for determing which events are clean, with respect to how they overlap other events, for inclusion in the analysis.',
    'q_th': 'The threshold in time points to determine when a quiet period has occurred.',
    'q_start_offset': 'The offset in time points to determine the marked start of a quiet period relative to the end of a preceding behavior.',
    'q_end_offset': 'The offset in time points to determine the marked end of a quiet period relative to the start of a succeeding behavior',
    'co_th': 'The threshold in time points used for marking real transitions.',
    'acc_behs': 'The set of acceptable behaviors transitioned into that could be included in the analysis',
    'acc_pre_behs': 'The set of acceptable behaviors transitioned from that could be included in the analysis',
    'pool_preceeding_turns': 'True if left and right preceding turns should be pooled.',
    'pool_succeeding_turns': 'True if succeeding left and right turns should be pooled.',
    'remove_st': 'True if self-transitions are removed before fitting models',
    'pre_ref_beh': 'The reference behavior for preceding behaviors',
    'ref_beh': 'The reference behavior for succeeding behaviors',
    'background': 'Background value for calculating DFF',
    'ep': 'Epsilon value for calculating DFF',
    'min_n_subjs': 'The min number of subjects we need to observe a transition for in order to include it in the model.',
    'min_n_events': 'Min number of events we must observe a preceding or succeeding behavior to include it in the analysis.',
    'window_type': 'The method for determining the alignment of the window of neural activity relative to behavior.',
    'window_offset': 'The offset in time points of the start of the window of neural activity relative to behavior.',
    'window_length': 'The length of the window in time points of neural activity analyzed.',
    'enforce_contained_events': 'True if only events with behaviors falling entirely within the window of analyzed neural activity should be included in the data used for model fitting.',
                           }


vis_yaml_fields = {'map_type': 'The type of statistics visualized - either original fit or mean compare.'}
# ======================================================================================================================
# Code goes here
# ======================================================================================================================

# Load pickle files containing fluorescence extraction and baseline calculation parameters
with open(f_extraction_params_file, 'rb') as f:
    f_extraction_params = pickle.load(f)

with open(baseline_calc_params_file, 'rb') as f:
    baseline_calc_params = pickle.load(f)


# Find all results
rs_files = glob.glob(str(Path(rs_folder) / '*.pkl'))
n_results = len(rs_files)

# Generate collection for all results
for f_i, f in enumerate(rs_files):

    if re.match('.*[0123456789].pkl', f) is not None:
        map_type = 'orig_fit'
        orig_fit_file = f
    else:
        map_type = 'mean_cmp'
        orig_fit_file = (re.match('(.*)_mean_cmp_stats.pkl', f)[1]) + '.pkl'

    image_folder = Path(f).parents[0] / (Path(f).stem + '_images')

    with open(orig_fit_file, 'rb') as f_h:
        mdl_fitting_results = pickle.load(f_h)

    # Determine the set of preceeding and succeeding behaviors
    preceding_behs = list(set(t[0] for t in mdl_fitting_results['beh_trans']))
    preceding_behs.sort()
    suceeding_behs = list(set(t[1] for t in mdl_fitting_results['beh_trans']))
    suceeding_behs.sort()

    # Form the collection
    params = [{'desc': 'f_extraction_params',
          'for_saving': f_extraction_params,
          'for_metadata': f_extraction_params, #f_extraction_params['roi_extract_opts'],
          'inc_params': f_extraction_yaml_fields},
          {'desc': 'baseline_calc_params',
          'for_saving': baseline_calc_params,
          'for_metadata': baseline_calc_params['baseline_calc_opts'],
          'inc_params': baseline_calc_yaml_fields},
          {'desc': 'mdl_fitting_params',
           'for_saving': mdl_fitting_results['ps'],
           'for_metadata': mdl_fitting_results['ps'],
           'inc_params': mdl_fitting_yaml_fields},
          {'desc': 'visualization_params',
           'for_saving': {'map_type': map_type},
           'for_metadata': {'map_type': map_type},
           'inc_params': vis_yaml_fields}]

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





