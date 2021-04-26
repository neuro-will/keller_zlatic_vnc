""" A script to generate collections after images and movies of maps have been generated.

This script should be called after make_spont_whole_brain_movies_and_images.

"""

# Give path to a file holding parameters used to extract fluorescece
f_extraction_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\extraction_params.pkl'

# Give path to a file holding parameters used to extract baselines
baseline_calc_params_file = r'K:\SV4\CW_17-08-23\L1-561nm-ROIMonitoring_20170823_145226.corrected\extracted\rois_1_5_5\long_baseline_extract_params.pkl'

# List those who can be contacted with questions about the collection
responsible = ['William Bishop <bishopw@hhmi.janelia.org>',
               'Andrew Champion <andrew.champion@gmail.com>']

# Provide a description of the collection.
description = ('In this analysis we look at results for the single EM specimen, analyzing spontaneous behavior ' +
               'transitions.  We sweep a window in time, to see how encoding changes relative to behavior onset.'
               'The key scripts to run the statistical tests used to produce these maps are fit_init_models.py ' +
               ' and find_vls_different_than_other_mean.py. The script ' +
               'make_spont_whole_brain_movies_and_images.py was then used to render the actual maps.')

# List hashes identify commits in git for the different pieces of code used to produce these results
git_hashes = {'janelia_core': '0cc52fb406b985b274d222ee16b05ba20365715d',
             'keller_zlatic_vnc': 'beda4ab71553e0a3693af0f37c853f5d2966fee2'}
