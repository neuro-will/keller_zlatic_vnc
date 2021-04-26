"""

Given a set of basic statistical results, which give coefficients for individual transitions,
this script will search for coefficients which are different than mean of all other coefficients. We do this
for each coefficient and record the p-values.

This will save results in a format that is conducive for working with existing plotting code.

"""

import multiprocessing as mp
from pathlib import Path
import pickle

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import test_for_diff_than_mean_vls

# ======================================================================================================================
# Parameters go here
# ======================================================================================================================

ps = dict()
ps['save_folder'] = r'\\dm11\bishoplab\projects\keller_vnc\results\single_subject_small_window_sweep'
ps['basic_rs_file'] = 'beh_stats_neg_18_3_turns_broken_out.pkl'

# ======================================================================================================================
# Code starts here
# ======================================================================================================================

if __name__ == '__main__':

    # ======================================================================================================================
    # Load basic results
    # ======================================================================================================================
    with open(Path(ps['save_folder']) / ps['basic_rs_file'], 'rb') as f:
        basic_rs = pickle.load(f)

    beh_trans = basic_rs['beh_trans']

    print('Done loading results from: ' + str(ps['basic_rs_file']))
    n_cpu = mp.cpu_count()
    with mp.Pool(n_cpu) as pool:
        all_mean_stats = pool.starmap(test_for_diff_than_mean_vls,
                                      [(s, beh_trans) for s in basic_rs['full_stats']])

    # ==================================================================================================================
    # Now save our results
    # ==================================================================================================================

    rs = {'ps': ps, 'full_stats': all_mean_stats, 'beh_trans': basic_rs['beh_trans']}

    save_folder = ps['save_folder']
    save_name = ps['basic_rs_file'].split('.')[0] + '_mean_cmp_stats.pkl'

    save_path = Path(save_folder) / save_name
    with open(save_path, 'wb') as f:
        pickle.dump(rs, f)

    print('Done.  Results saved to: ' + str(save_path))

