# Python script to test for stimulus dependence across the whole brain

import multiprocessing as mp
import numpy as np
import os
from pathlib import Path

from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import whole_brain_stimulus_dep_testing
from keller_zlatic_vnc.whole_brain.whole_brain_stat_functions import make_whole_brain_videos_and_max_projs

# ==============================================================================
# Parameters go here: anything that is a list are values we will loop through
# ==============================================================================

# The data files we want to run tests on
data_files = [Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_1_5_5_long_bl.pkl')]#,
             # Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_1_5_5.pkl'),
             # Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_4_20_20_long_bl.pkl'),
             # Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_4_20_20.pkl'),
             # Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_12_60_60_long_bl.pkl'),
             # Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_12_60_60.pkl')]

#data_files = [Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6\dff_4_20_20_long_bl.pkl')]

# The corresponding roi groups the results in data_files are for
roi_groups = ['rois_1_5_5']#,
             # 'rois_1_5_5',
             # 'rois_4_20_20',
             # 'rois_4_20_20',
             # 'rois_12_60_60',
             # 'rois_12_60_60']

#roi_groups = ['rois_4_20_20']

# Manipulation types we want to consider
manip_types = ['both'] #['A4', 'A9', 'both']

# Parameters for tests
n_perms = 10000

# Location of overlay files
# Specify where we find overlay files
overlay_files = [r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\horz_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\cor_mean.png',
                 r'\\dm11\bishoplab\projects\keller_vnc\data\overlays\sag_mean.png']

save_folder = Path(r'A:\projects\keller_vnc\results\whole_brain_stats\v6')

# Specify max number of cores we can use
max_n_cpu = 40

# Define helper functions
def stim_dep_analysis(kwargs, pool):
    stat_kwargs = kwargs[0]
    stat_kwargs['pool'] = pool
    plot_kwargs = kwargs[1]

    results_file = whole_brain_stimulus_dep_testing(**stat_kwargs)

    make_whole_brain_videos_and_max_projs(results_file=results_file, overlay_files=overlay_files,
                                          save_supp_str=plot_kwargs['save_str'],
                                          roi_group=plot_kwargs['roi_group'],
                                          gen_mean_tiff=False, gen_mean_movie=False,
                                          gen_coef_movies=True, gen_coef_tiffs=False,
                                          gen_p_value_movies=True, gen_p_value_tiffs=False,
                                          gen_filtered_coef_movies=False, gen_filtered_coef_tiffs=False,
                                          gen_combined_tiffs=False, gen_combined_movies=True,
                                          gen_combined_projs=True, gen_uber_movies=True)


# Generate all possible parameter combinations
param_vls = []
for data_file, roi_group in zip(data_files, roi_groups):
    for manip_type in manip_types:

        data_file_stem = Path(data_file).stem
        cur_save_folder = Path(save_folder) / data_file_stem

        if not os.path.isdir(cur_save_folder):
            os.mkdir(cur_save_folder)

        desc_str = 'mt_' + manip_type

        param_vls.append(({'data_file': data_file,
                           'manip_type': manip_type,
                           'save_folder': cur_save_folder,
                           'save_str': desc_str,
                           'n_perms': n_perms},
                            {'roi_group': roi_group,
                              'save_str': desc_str + '_' + data_file_stem}))

if __name__ == '__main__':
    n_combs = len(param_vls)
    # See how many processors we have available
    cpu_count = mp.cpu_count()
    n_used_cpus = cpu_count
    if n_used_cpus > max_n_cpu:
        n_used_cpus = max_n_cpu
    print('Processing ' + str(n_combs) + ' combinations of parameters with ' + str(n_used_cpus) + ' processes.')

    pool = mp.Pool(n_used_cpus)

    ## Perform the analyses
    for param_vls_i in param_vls:
        stim_dep_analysis(param_vls_i, pool)

