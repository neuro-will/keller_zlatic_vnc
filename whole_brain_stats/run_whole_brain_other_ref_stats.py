# Python script to call the function whole_brain_other_ref_testing and
# then make videos and tiffs of results for different combinations of parameters

import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
import pickle

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import whole_brain_other_ref_testing
from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs

# ==============================================================================
# Parameters go here: anything that is a list are values we will loop through
# ==============================================================================
# The data files we want to run tests on
data_files = [Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_4_20_20.pkl'),
              Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_4_20_20_long_bl.pkl'),
             Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_2_10_10.pkl'),
             Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_2_10_10_long_bl.pkl'),
              Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_1_5_5.pkl'),
             Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8\dff_1_5_5_long_bl.pkl')]

# The corresponding roi groups the results in data_files are for
roi_groups = ['rois_4_20_20',
              'rois_4_20_20',
              'rois_2_10_10',
              'rois_2_10_10',
              'rois_1_5_5',
              'rois_1_5_5']

# The types of tests we want to run
test_types = ['prediction_dependence', 'state_dependence', 'decision_dependence', 'before_reporting', 'after_reporting']

# Cut off times we want to consider
cut_off_times = [3.231, 5.4997, 17.4523]

# Manipulation types we want to consider
manip_types = ['A4', 'A9', 'both']

save_folder = Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v8')

min_n_subjects_per_beh = 3

beh_ref = 'Q'

alpha = .05

max_n_cpu = 10

# Location of overlay files
# Specify where we find overlay files
overlay_files = [r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\horz_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\cor_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\sag_mean.png']


# Define helper functions
def single_ref_analysis(kwargs):
    stat_kwargs = kwargs[0]
    plot_kwargs = kwargs[1]

    # See if we have already completed results for this file
    status_file_name = '._status_' + stat_kwargs['save_str'] + '_' + Path(stat_kwargs['data_file']).stem  + '.pkl'
    status_file_path = stat_kwargs['save_folder'] / status_file_name
    if os.path.exists(status_file_path):
        return
    else:
        # Process results if we need to
        results_file = whole_brain_other_ref_testing(**stat_kwargs)

        make_whole_brain_videos_and_max_projs(results_file=results_file, overlay_files=overlay_files,
                                              save_supp_str=plot_kwargs['save_str'],
                                              roi_group=plot_kwargs['roi_group'],
                                              gen_mean_tiff=False, gen_mean_movie=False,
                                              gen_coef_movies=True, gen_coef_tiffs=False,
                                              gen_p_value_movies=True, gen_p_value_tiffs=False,
                                              gen_filtered_coef_movies=False, gen_filtered_coef_tiffs=False,
                                              gen_combined_tiffs=False, gen_combined_movies=True,
                                              gen_combined_projs=True, gen_uber_movies=True)

        # Save a small marker file noting that we are done
        with open(status_file_path, 'wb') as f:
            pickle.dump({'done': True}, f)


# Run everything in parallel
if __name__ == '__main__':
    # ==============================================================================
    # Generate all possible parameter combinations

    param_vls = []
    for data_file, roi_group in zip(data_files, roi_groups):
        for test_type in test_types:
            for cut_off_time in cut_off_times:
                for manip_type in manip_types:

                    data_file_stem = Path(data_file).stem
                    cur_save_folder = Path(save_folder) / data_file_stem

                    if not os.path.isdir(cur_save_folder):
                        os.mkdir(cur_save_folder)

                    desc_str = (test_type + '_ref_O' + '_cut_off_time_' + str(cut_off_time) +
                                '_mt_' + manip_type)

                    desc_str = desc_str.replace('.', '_')

                    param_vls.append(({'data_file': data_file,
                                      'test_type': test_type,
                                      'cut_off_time': cut_off_time,
                                      'manip_type': manip_type,
                                      'save_folder': cur_save_folder,
                                      'save_str': desc_str,
                                      'min_n_subjects_per_beh': min_n_subjects_per_beh,
                                      'beh_ref': beh_ref,
                                      'alpha': alpha},
                                      {'roi_group': roi_group,
                                       'save_str': desc_str + '_' + data_file_stem}))

    n_combs = len(param_vls)

    # ==============================================================================
    # See how many processors we have available
    cpu_count = mp.cpu_count()
    n_used_cpus = np.min([cpu_count, n_combs])
    if n_used_cpus > max_n_cpu:
        n_used_cpus = max_n_cpu
    print('Processing ' + str(n_combs) + ' combinations of parameters with ' + str(n_used_cpus) + ' processes.')

    # Perform the analyses
    #for param_vls_i in param_vls:
    #    single_ref_analysis(param_vls_i)

    pool = mp.Pool(n_used_cpus)
    pool.map(single_ref_analysis, param_vls)