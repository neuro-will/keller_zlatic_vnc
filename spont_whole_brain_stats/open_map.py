## A script for finding and opening a particular map in a merged collection
import os
from pathlib import Path

pre_beh = 'G'
suc_beh = 'B'

map_type = 'coef_p_vls_comb'
map_ext = '.mp4'

options = {'clean_event_def': 'decision',
           'pool_succeeding_turns': 'False',
           'window_length': '3',
           'window_offset': '0'}

coll_folder = r'/Volumes/bishoplab/projects/keller_vnc/results/single_subject/spont_window_sweep/spont_window_sweep'

## Open the file
f_name = pre_beh + '_' + suc_beh + '__' + map_type + '_'

for key, vl in options.items():
    f_name += key + '_' + vl + '_'

f_name = f_name[0:-1] + map_ext
f_path = Path(coll_folder) / f_name

os.system('open ' + str(f_path))
