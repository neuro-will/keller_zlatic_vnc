# Python script to call the function whole_brain_state_functions.py

import argparse
from pathlib import Path

from whole_brain_stat_functions import whole_brain_single_ref_testing


parser = argparse.ArgumentParser(description='Run whole brain single reference tests given particular settings.')

parser.add_argument('data_file', type=str)
parser.add_argument('test_type', type=str)
parser.add_argument('cut_off_time', type=float)
parser.add_argument('manip_type', type=str)
parser.add_argument('save_folder', type=str)
parser.add_argument('save_str', type=str)
parser.add_argument('-min_n_subjects_per_beh', type=int, default=3)
parser.add_argument('-beh_ref', type=str, default='Q')
parser.add_argument('-alpha', type=float, default=.05)

args = parser.parse_args()

data_file = Path(args.data_file)
test_type = args.test_type
cut_off_time = args.cut_off_time
manip_type = args.manip_type
save_folder = Path(args.save_folder)
save_str = args.save_str
min_n_subjects_per_beh = args.min_n_subjects_per_beh
beh_ref = args.beh_ref
alpha = args.alpha


whole_brain_single_ref_testing(data_file=data_file, test_type=test_type, cut_off_time=cut_off_time,
                               manip_type=manip_type, save_folder=save_folder, save_str=save_str,
                               min_n_subjects_per_beh=min_n_subjects_per_beh, beh_ref=beh_ref, alpha=alpha)

make_whole_brain_videos_and_max_projs(results_file=Path('/groups/bishop/bishoplab/projects/keller_vnc/results/whole_brain_stats/v3/after_reporting_other_ref_A4_A9_2020_10_05_12_45_20_805624.pkl'),
                                      save_supp_str='testing', roi_group='rois_2_10_10')