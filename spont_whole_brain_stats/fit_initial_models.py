""" A python script for fitting models to spontaneous data.  This script wraps around the function fit_init_models for
fitting models to spontaneous data.  The main purposes of this script are to:

1) Allow models to be fit with different combinations of parameters

2) Provide for a picking back up from where it left off if it were to crash in the middle of a run (an important
consideration when fitting models with many different combinations of parameters.

See the function fit_init_models for the specifics of how models are fit.

"""

import glob
from pathlib import Path
import pickle

from janelia_core.utils.data_saving import append_ts

from keller_zlatic_vnc.utils import form_combinations_from_dict
from keller_zlatic_vnc.whole_brain.spontaneous import fit_init_models

# ======================================================================================================================
# Parameters go here.
#
# Here we provide values that we use for fitting.  To specify that we should sweep through values for a parameter,
# provide those values in a list.  If we need to provide a single value, which is itself a list, then to ensure the
# code recognizes the list as a single value, enclose it an another list.  The code will then interpret this as
# a list of parameters of length 1, with a single value which is itself a list.
# ======================================================================================================================

base_ps = dict()

# Folders containing a4 and a9 annotation data
base_ps['annot_folders'] = [[r'/groups/bishop/bishoplab/projects/keller_vnc/data/full_annotations/em_volume_behavior_csv']]

# File containing locations to registered volumes
#ps['volume_loc_file'] = r'\\dm11\bishoplab\projects\keller_vnc\data\experiment_data_locations.xlsx'
base_ps['volume_loc_file'] = r'/groups/bishop/bishoplab/projects/keller_vnc/data/EM_volume_experiment_data_locations.xlsx'

# List subjects we do not want to include in the analysis
base_ps['exclude_subjs'] = set(['CW_17-11-06-L2'])

# Subfolder containing the dataset for each subject
base_ps['dataset_folder'] = 'extracted'

# Base folder where datasets are stored
base_ps['dataset_base_folder'] = r'K:\\SV4'

# The defintion we use for clean events
base_ps['clean_event_def'] = 'dj'  # 'dj' or 'po'

# Specify the threshold we use (in number of stacks) to determine when a quiet transition has occurred
base_ps['q_th'] = 55

# Specify the cut off threshold we use (in number of stacks) to determine when a real transition has occurred
base_ps['co_th'] = 3

# Specify the set of behaviors transitioned into for events we analyze
base_ps['behs'] = [['B', 'F', 'TL', 'TR', 'H']]

# Specify the preceding behaviors for events we analyze
base_ps['pre_behs'] = [['B', 'F', 'TL', 'TR', 'H', 'U']]

# True if we want to pool preceeding behaviors
base_ps['pool_preceeding_behaviors'] = True

# True if we want to pool preceding left and right turns into one category (only applies if pool_preceeding_behaviors
# is false)
base_ps['pool_preceeding_turns'] = True

# True if we want to pool succeeding left and right turns into one category
base_ps['pool_succeeding_turns'] = [False, True]

base_ps['remove_st'] = True

# Data to calculate Delta F/F for in each dataset
base_ps['f_ts_str'] = 'f_1_5_5'
base_ps['bl_ts_str'] = 'bl_1_5_5_long'

# Parameters for calculating dff
base_ps['background'] = 100
base_ps['ep'] = 20

# Min number of subjects we must observe a transition in for us to analyze it
base_ps['min_n_subjs'] = 1

# Min number of events we must observe a transition in for us to analyze it
base_ps['min_n_events'] = 5

# Alpha value for thresholding p-values when calculating stats - this is not used in producing the final maps
base_ps['alpha'] = .05

# Specify the window we pull dff from
base_ps['window_type'] = 'start_locked'  # 'whole_event' or 'start_locked'

# If we are using a window locked to event start or stop, we give the relative offset and window length here
base_ps['window_offset'] = [0, 3, 6, 9, 12, 15, 18]
base_ps['window_length'] = [3, 6]

# Specify if we only consider events where the extracted dff window is entirely contained within the event
base_ps['enforce_contained_events'] = False

# Specify folder where we should save results
base_ps['save_folder'] = r'/groups/bishop/bishoplab/projects/keller_vnc/results/single_subject/spont_window_sweep_v2/ind_collections'

# Specify a string for saving results with - results for each set of parameters will be saved in files with this string
# and a unique number (generated from the time) appended
base_ps['save_str'] = 'sweep'

# ======================================================================================================================
# Generate dictionaries for all combinations of parameters
# ======================================================================================================================

comb_ps = form_combinations_from_dict(base_ps)

# ======================================================================================================================
# Fit models for all combinations of parameters
# ======================================================================================================================

# Get a list of saved "pick up" files, saving the parameters for which we already have results - we can consult these
# files to "pick up" where we left off in a crash occurs in the running of this script
existing_pickup_files = glob.glob(str(Path(base_ps['save_folder']) / '.*.pkl'))
n_existing_pickup_files = len(existing_pickup_files)
existing_param_dicts = [None]*n_existing_pickup_files
for f_i, pu_file in enumerate(existing_pickup_files):
    with open(pu_file, 'rb') as f:
        existing_param_dicts[f_i] = pickle.load(f)
        del existing_param_dicts[f_i]['save_name']  # Remove save_name field, since this not in the ps dictionaries we
                                                    # check against

for c_i, ps in enumerate(comb_ps):
    print('===========================================================================================================')
    print('Performing analysis ' + str(c_i + 1) + ' of ' + str(len(comb_ps)) + '.')
    print('===========================================================================================================')

    # Before actually running results for this set of parameters, see if a set of results already exists in the save
    # folder.

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

        fit_init_models(ps)

        # Save the parameter dictionary as a seperate file - this is so we can pick back up if this script crashes
        pickup_file = Path(ps['save_folder']) / ('.' + ps['save_name'])
        with open(pickup_file, 'wb') as f:
            pickle.dump(ps, f)
        print('Analysis complete.  Results saved to: ' + str(Path(ps['save_folder']) / ps['save_name']))
    else:
        print('Discovered existing results.  ')


