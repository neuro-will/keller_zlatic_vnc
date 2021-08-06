""" A script to generate collections after images and movies of maps have been generated.

This script should be called after make_spont_whole_brain_movies_and_images.

Note a couple of "odd" things:

1) The user needs to supply a path to a file with the parameters that were used for fluorescence extraction.  In
reality, these paramters were saved with each specimen we extracted fluorescence from, so there are many files that
could be provided.  The user has to only supply one.  Note that these will be for a single setting of voxel sizes.

2) The user also has to supply a path to a file with the parameters used to exract baselines.  The same caveats and
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
rs_folder = r'\\dm11\bishoplab\projects\keller_vnc\results\single_subject\ind_trans_window_sweep\ind_collections'

# Give path to a file holding parameters used to extract fluorescece
f_extraction_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\extraction_params.pkl'

# Give path to a file holding parameters used to extract baselines
baseline_calc_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\long_baseline_extract_params.pkl'

# List those who can be contacted with questions about the collection
responsible = ['William Bishop <bishopw@hhmi.janelia.org>',
               'Andrew Champion <andrew.champion@gmail.com>']

# Provide a description of the collection.
description = ('In this analysis we look at results for the single EM specimen, analyzing spontaneous behavior ' +
               'for specific transitions. '  +
               'The key scripts to run the statistical tests used to produce these maps are fit_init_models.py ' +
               ' and find_vls_different_than_other_mean.py. The script ' +
               'make_spont_whole_brain_movies_and_images.py was then used to render the actual maps.')

# List hashes identify commits in git for the different pieces of code used to produce these results
git_hashes = {'janelia_core': 'c7f1c61635c5f32a513b43d0c0e0810a61f69007',
             'keller_zlatic_vnc': 'b5fdcd74c2f4a170d9323301f97234077eb2a317'}

# List the parameters that should be included in the metadata file, with comments that should also be included
f_extraction_yaml_fields = {'voxel_size_per_dim': 'Number of voxels in each dimension of a supervoxel.'}

baseline_calc_yaml_fields = {'window_length': 'Length of window used for baseline calculation.',
                             'filter_start': 'Initial offset, relative to first data point, of window used for baseline calculations.',
                             'write_offset': "Offset between first point in window and the point the filtered output is assigned to.",
                             'p': 'The particular percentile used for percentile filtering.'}

mdl_fitting_yaml_fields = {'co_th': 'The threshold in time points used for marking real transitions.',
                           'q_th': 'The threshold in time points used for determining if a behavior is preceeded by quiet.',
                           'enforce_contained_events': 'True if only events with behaviors falling entirely within the window of analyzed neural activity should be included in the data used for model fitting.',
                           'pool_preceeding_behaviors': 'True if all preceeding behaviors should be pooled.',
                           'pool_preceeding_turns': 'True if left and right preceeding turns should be pooled.',
                           'pool_succeeding_turns': 'True if succeeding left and right turns should be pooled.',
                           'clean_event_def': 'The criteria used for determing which events are clean, with respect to how they overlap other events, for inclusion in the analysis.',
                           'behs': 'Filter applied so that only events with behaviors transitioned into in this set are included in the analysis.',
                           'pre_behs': 'Filter applied so that only events with preceeding behaviors in this set are included in the analysis.',
                           'remove_st': 'True if we remove any events with transitions to and from the same behavior from the analysis',
                           'min_n_subjs': 'The min number of subjects we need to observe a transition for in order to include it in the model.',
                           'min_n_events': 'The min number of times we need to observe a transition (across all subejcts) to include it in the model.',
                           'window_type': 'The method for determining the alignment of the window of neural activity relative to behavior.',
                           'window_offset': 'The offset in time points of the start of the window of neural activity relative to behavior.',
                           'window_length': 'The length of the window in time points of neural activity analyzed.'
                           }

mean_cmp_yaml_fields = {}

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
rs_files = [f for f in rs_files if re.match('.*mean_cmp_stats.pkl', f) is None]
n_results = len(rs_files)

for f_i, f in enumerate(rs_files):
    mdl_fitting_params_file = f
    mean_cmp_params_file = Path(f).parents[0] / (Path(f).stem + '_mean_cmp_stats.pkl')
    image_folder = Path(f).parents[0] / (Path(f).stem + '_mean_cmp_stats_images')

    with open(mdl_fitting_params_file, 'rb') as f:
        mdl_fitting_results = pickle.load(f)

    with open(mean_cmp_params_file, 'rb') as f:
        mean_cmp_results = pickle.load(f)

    # Determine the set of preceeding and succeeding behaviors
    preceding_behs = list(set(t[0] for t in mean_cmp_results['beh_trans']))
    preceding_behs.sort()
    suceeding_behs = list(set(t[1] for t in mean_cmp_results['beh_trans']))
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
           'inc_params': mdl_fitting_yaml_fields},
          {'desc': 'mean_cmp_params',
           'for_saving': mean_cmp_results['ps'],
           'for_metadata': mean_cmp_results['ps'],
           'inc_params': mean_cmp_yaml_fields}]

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





