""" A python script to look at the response to pain (stimulus) of different voxels across the brain of a single subject

"""

import glob
from pathlib import Path
import pickle

from janelia_core.utils.data_saving import append_ts

from keller_zlatic_vnc.utils import form_combinations_from_dict
from keller_zlatic_vnc.whole_brain.pain import single_subject_pain_stats

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

base_ps = dict()

# Folders containing a4 and a9 annotation data
base_ps['annot_folders'] = [[r'\\dm11\bishoplab\projects\keller_vnc\data\full_annotations\em_volume_behavior_csv']]

# Subject we analyze
base_ps['analyze_subj'] = 'CW_18-02-15-L1'

# File containing locations to registered volumes
base_ps['volume_loc_file'] = r'\\dm11\bishoplab\projects\keller_vnc\data\EM_volume_experiment_data_locations.xlsx'

# Subfolder containing the dataset for each subject
base_ps['dataset_folder'] = 'extracted'

# Base folder where datasets are stored
base_ps['dataset_base_folder'] = r'K:\\SV4'

# Data to calculate Delta F/F for in each dataset
base_ps['f_ts_str'] = 'f_1_5_5'
base_ps['bl_ts_str'] = 'bl_1_5_5_long'

# Parameters for calculating dff
base_ps['background'] = 100
base_ps['ep'] = 20

# Parameters for filtering events based on stimulus duration
base_ps['min_stim_dur'] = 5
base_ps['max_stim_dur'] = 15

# Length of window we pull dff in from before the stimulus
base_ps['n_before_tm_pts'] = 3

# Specify if we align the after window to the end of the stimulus or the beginning of the stimulus, can be
# either 'start' or 'end'
base_ps['after_aligned'] = ['start']

# Offset from the start of the window for dff after the event and the last stimulus time point.  An offset of 0,
# means the first time point in the window will be the last time point the stimulus was delevered
base_ps['after_offset'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Length of window we pull dff in from after the stimulus
base_ps['n_after_tm_pts'] = [1, 3]

# Folder where we should save results
base_ps['save_folder'] = r'A:\projects\keller_vnc\results\single_subject_pain_maps_v2'

# String to save with file names
base_ps['save_str'] = 'pain'


# ======================================================================================================================
# Generate dictionaries for all combinations of parameters
# ======================================================================================================================

comb_ps = form_combinations_from_dict(base_ps)

# ======================================================================================================================
# Get a list of saved "pick up" files, saving the parameters for which we already have results - we can consult these
# files to "pick up" where we left off in a crash occurs in the running of this script
# ======================================================================================================================
existing_pickup_files = glob.glob(str(Path(base_ps['save_folder']) / '.*.pkl'))
n_existing_pickup_files = len(existing_pickup_files)
existing_param_dicts = [None]*n_existing_pickup_files
for f_i, pu_file in enumerate(existing_pickup_files):
    with open(pu_file, 'rb') as f:
        existing_param_dicts[f_i] = pickle.load(f)
        del existing_param_dicts[f_i]['save_name']  # Remove save_name field, since this not in the ps dictionaries we
                                                    # check against

# ======================================================================================================================
# Fit models for all combinations of parameters
# ======================================================================================================================
for c_i, ps in enumerate(comb_ps):
    print('===========================================================================================================')
    print('Performing analysis ' + str(c_i + 1) + ' of ' + str(len(comb_ps)) + '.')
    print('===========================================================================================================')

    # Remove the save_str field from ps here, since we don't want to save it (we replace with with save_name)
    save_str = ps['save_str']
    del ps['save_str']

    results_exist = False
    for d in existing_param_dicts:
        if ps == d:
            results_exist = True
            break

    if not results_exist:
        ps['save_name'] = append_ts(save_str, no_underscores=True) + '.pkl'

        # Actually do the analysis here
        single_subject_pain_stats(**ps)

        # Save the parameter dictionary as a separate file - this is so we can pick back up if this script crashes
        pickup_file = Path(ps['save_folder']) / ('.' + ps['save_name'])
        with open(pickup_file, 'wb') as f:
            pickle.dump(ps, f)
        print('Analysis complete.  Results saved to: ' + str(Path(ps['save_folder']) / ps['save_name']))
    else:
        print('Discovered existing results.  ')
