from pathlib import Path
import subprocess


base_dir = r'/Volumes/bishoplab/projects/keller_vnc/results/whole_brain_stats/v10_organized'

# Specify which results we want to load
results = [{'test_type': 'before_reporting',
           'behavior': 'T',
           'ref': 'O',
           'cut_off_time': 3.231, #3.231, #17.4523, #5.4997,
           'manipulation_type': 'A4',
           'preprocessing_str': 'dff_1_5_5_long_bl'}]


# Load the results
for r in results:

    # Form the path to the file we want to open
    ct_str = str(r['cut_off_time']).replace('.', '_')
    f_name = (r['behavior'] + '_' + r['test_type'] + '_ref_' + r['ref'] + '_cut_off_time_'  + ct_str +
              '_mt_' + r['manipulation_type'] + '_' + r['preprocessing_str'] + '_coef_p_vls_comb.mp4')
    f_path = Path(base_dir) / r['test_type'] / r['behavior'] / f_name

    # Open the file in quicktime
    command = ('open', '-a', 'Quicktime Player', str(f_path))
    subprocess.check_call(command)






