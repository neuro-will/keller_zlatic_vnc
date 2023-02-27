""" A python script for fitting models to closed-loop data.  The main purposes of this script are to:

1) Allow models to be fit with different combinations of parameters

2) Provide for a picking back up from where it left off if it were to crash in the middle of a run (an important
consideration when fitting models with many different combinations of parameters.

See the function fit_init_models for the specifics of how models are fit.

"""

import glob
from pathlib import Path
import pickle

from janelia_core.utils.data_saving import append_ts
from keller_zlatic_vnc.whole_brain.closed_loop import fit_init_models
from keller_zlatic_vnc.utils import form_combinations_from_dict

# ======================================================================================================================
# Parameters go here.
#
# Here we provide values that we use for fitting.  To specify that we should sweep through values for a parameter,
# provide those values in a list.  If we need to provide a single value, which is itself a list, then the first
# value of the list should be the string 'ds' to indicate that we should not split that list.
# ======================================================================================================================

base_ps = dict()

base_ps['processed_data_file'] = r'Z:\bishoplab\projects\keller_drive\keller_vnc\results\new_whole_brain_stats\1_5_5\state_dep_1_5_5.pkl'

# The manipulation targer we want to analyze events for.  Either 'A4', 'A9' or None (which indicates pooling).
base_ps['manipulation_tgt'] = None

# True if we are suppose to pool left and right turns
base_ps['pool_turns'] = [True, False]

# The set of before and after behaviors we want to fit models to.  If None, all behaviors will be used.
base_ps['behs'] = ['ds', 'Q', 'TC', 'TL', 'TR',  'B', 'F', 'H']

# Mininum number of subjects we must see a preceeding behavior in to include in the analysis
base_ps['min_n_pre_subjs'] = 3

# Mininum number of subjects we must see a succeeding behavior in to include in the analysis
base_ps['min_n_succ_subjs'] = 3

# The reference behavior for modeling
base_ps['ref_beh'] = 'Q'

# The significance level we reject individual null hypotheses at at
base_ps['ind_alpha'] = .05

# Specify folder where we should save results
base_ps['save_folder'] = r'Z:\bishoplab\projects\keller_drive\keller_vnc\results\new_whole_brain_stats\1_5_5\state_dep'

# Specify a string for saving results with - results for each set of parameters will be saved in files with this string
# and a unique number (generated from the time) appended
base_ps['save_str'] = 'init_fit'

# ======================================================================================================================
# Generate dictionaries for all combinations of parameters
# ======================================================================================================================

comb_ps = form_combinations_from_dict(base_ps)

# ======================================================================================================================
# Fit models for all combinations of parameters
# ======================================================================================================================

if __name__ == '__main__':

    # Get a list of saved "pick up" files, saving the parameters for which we already have results - we can consult
    # these files to "pick up" where we left off in a crash occurs in the running of this script
    existing_pickup_files = glob.glob(str(Path(base_ps['save_folder']) / '.*.pkl'))
    n_existing_pickup_files = len(existing_pickup_files)
    existing_param_dicts = [None]*n_existing_pickup_files
    for f_i, pu_file in enumerate(existing_pickup_files):
        with open(pu_file, 'rb') as f:
            existing_param_dicts[f_i] = pickle.load(f)
            # Remove save_name field, since this not in the ps dictionaries we check against
            del existing_param_dicts[f_i]['save_name']

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

            # Save the parameter dictionary as a separate file - this is so we can pick back up if this script crashes
            pickup_file = Path(ps['save_folder']) / ('.' + ps['save_name'])
            with open(pickup_file, 'wb') as f:
                pickle.dump(ps, f)
            print('Analysis complete.  Results saved to: ' + str(Path(ps['save_folder']) / ps['save_name']))
        else:
            print('Discovered existing results.  ')
