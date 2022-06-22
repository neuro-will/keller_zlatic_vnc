## A script for finding and opening a particular map in a merged collection
import os
from pathlib import Path

# Specify location of collection 
coll_folder = r'/Volumes/bishoplab/projects/keller_vnc/results/single_subject/new_model_maps_v0/all_supervoxels_v0_collection'
#coll_folder = r'/Volumes/bishoplab/projects/keller_vnc/results/single_subject/new_model_maps_v1/brain_only_merged'

# Specify the type of behavior we want to look at
beh_str = 'beh_TC'

# Specify the options for the map we want to look for
options = {'map_type': 'orig_fit', 
           'window_length': 1, 
           'window_offset': 1,
           'pool_succeeding_turns': True, 
           'pool_preceeding_turns': True, 
	   'co_th': 3,
	   'q_start_offset': 1}

## Open the file
f_name = beh_str + '__coef_p_vls_comb_'

for key, vl in options.items():
    f_name += key + '_' + str(vl) + '_'

f_name = f_name[0:-1] + '.mp4'
f_path = Path(coll_folder) / f_name

os.system('open ' + str(f_path))
