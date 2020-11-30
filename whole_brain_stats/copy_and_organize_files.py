# This is a script for copying images and movies of processed results, while organizing them in a new
# directory structure

import glob
import os
import pathlib
import re
import shutil

# ======================================================================================================================
#  Parameters go here

# Directories where we will pull images from - these should have subfolders for each separate set of test setting
base_dirs = [r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_4_20_20_long_bl',
             r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_2_10_10_long_bl',
             r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_1_5_5_long_bl',
             r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_4_20_20',
             r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_2_10_10',
             r'A:\projects\keller_vnc\results\whole_brain_stats\v10\dff_1_5_5']

# Specify where we save new results
tgt_dir = r'A:\projects\keller_vnc\results\whole_brain_stats\v10_organized'

# List to type of tests we want to copy and organize
test_types = ['before_reporting', 'after_reporting']

# List to type of suffix of files we want to copy
file_types = ['comb.mp4', '.tiff', '.png']

# ======================================================================================================================
#  Create target folder if we need to

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

# ======================================================================================================================
# Process files

test_type_str = '|'.join(test_types)

for b_dir in base_dirs:
    sub_folders = [d for d in os.listdir(b_dir) if os.path.isdir(os.path.join(b_dir, d))]
    for sub_folder in sub_folders:
        sub_folder = pathlib.Path(b_dir) / sub_folder
        for file_type in file_types:
            type_files = glob.glob(str(sub_folder / ('*' + file_type)))
            for file in type_files:
                file_name = pathlib.Path(file).name
                test_match = re.match('(.*)_(' + test_type_str + ')_.*', file_name)
                if test_match is not None:
                    beh = test_match.group(1)
                    test_type = test_match.group(2)

                    new_folder = pathlib.Path(tgt_dir) / test_type / beh

                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)

                    copy_src_path = sub_folder / file_name
                    copy_tgt_path = new_folder / file_name

                    #if os.path.exists(copy_tgt_path):
                    #    raise(RuntimeError('Caught existing file: ' + str(copy_tgt_path)))
                    #else:
                    shutil.copyfile(copy_src_path, copy_tgt_path)




