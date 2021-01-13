""" Contains tools for reading in and converting data. """

import copy
import os.path
from typing import Sequence, Tuple, Union
import pathlib
import pickle
import re

import numpy as np
import pandas as pd

from janelia_core.dataprocessing.dataset import ROIDataset
from janelia_core.dataprocessing.roi import ROI
from janelia_core.fileio.data_handlers import NDArrayHandler
from janelia_core.fileio.exp_reader import find_images
from janelia_core.utils.searching import dict_find

# Dictionary defining the codes we will use for behavior and how these are coded in the original data
BEHAVIOR_CODES = {'Q': ['Quiet', 'quiet'],
                  'F': ['Forward', 'forward'],
                  'B': ['Backward', 'backward'],
                  'H': ['Hunch', 'hunch', ' hunch'],
                  'T': ['Turn', 'turn'],
                  'O': ['Other', 'other', 'others'],
                  'P': ['Back Hunch', 'back hunch'],
                  'TL': ['turn (left turn)', 'Turn (left turn)'],
                  'TR': ['turn (right turn)']}

FULL_ANNOT_BEH_CODES = {'F': ['fw'],
                        'B': ['bw'],
                        'S': ['stim'],
                        'H': ['hunch'],
                        'T': ['turn'],
                        'O': ['other'],
                        'P': ['HP'],
                        'TL': ['left turn'],
                        'TR': ['right turn']}

# Dictionary defining A00c segment codes
A00C_SEG_CODES = {1: 'antL',
                  2: 'antR',
                  3: 'midL',
                  4: 'midR',
                  5: 'postL',
                  6: 'postR'}

# Dictionary defining Handle segment codes
HANDLE_SEG_CODES = {1: 'T1',
                    2: 'T2',
                    3: 'T3',
                    4: 'A1',
                    5: 'A2',
                    6: 'A3',
                    7: 'A4',
                    8: 'A5',
                    9: 'A6',
                    10: 'A7',
                    11: 'A8',
                    12: 'A9'}

# Dictionary defining basin segment codes
BASIN_SEG_CODES = {1: 'T1 L+R',
                   2: 'T2 L+R',
                   3: 'T3 L+R',
                   4: 'A1 L+R',
                   5: 'A2 L+R',
                   6: 'A3 L+R',
                   7: 'A4 L+R',
                   8: 'A5 L+R',
                   9: 'A6 L+R',
                   10: 'A7 L+R',
                   11: 'A8 L+R',
                   12: 'A9 L+R'}


def combine_turns(tbl: pd.DataFrame):
    """ Combines annotations for TL and TR into a single T annotation in a transition table.

    This function assumes that behavior in the transition table has been recoded to use standard abbreivations.

    Args:

        tbl: The table to with behavior to recode

    Returns:
        Nothing.  The table will be modified in place.
    """
    tbl['beh_before'][tbl['beh_before'] == 'TL'] = 'T'
    tbl['beh_before'][tbl['beh_before'] == 'TR'] = 'T'
    tbl['beh_after'][tbl['beh_after'] == 'TL'] = 'T'
    tbl['beh_after'][tbl['beh_after'] == 'TR'] = 'T'


def extract_transitions(raw_trans_table: pd.DataFrame, cutoff_time:float = np.inf) -> Tuple[dict, pd.DataFrame]:
    """ Calculates new behavior transitions, using a cutoff time for behaviors after the stimulus.

    This function assumes that behavior in the input table has been recoded to use standard abbreviations.

    Args:
        raw_trans_table: The table of raw transition information; as loaded with read_raw_transitions_from_excel

        cutoff_time: The cutoff time to use to calculate transitions.

    Returns:
        trans: A dictionary with transitions.  Keys correspond to subject ids.  Values are lists of transitions, in the
        order they were listed in the raw_trans_table. Each transition is a tuple of the form (before_beh, after_beh).

        new_table: A copy of the original transition table, with updated behaviors after the manipulation
    """

    # Get list of unique specimen ids
    spec_ids = raw_trans_table['subject_id'].unique()

    # Extract transitions
    table_copy = copy.deepcopy(raw_trans_table)

    # Relabel behaviors after the stimulus that took too long to occur as quiet
    new_quiet_inds = table_copy['Trans Time'] > cutoff_time
    table_copy.loc[new_quiet_inds, 'beh_after'] = 'Q'

    # Extract transitions for each specimen
    trans = dict()
    for spec_id in spec_ids:
        spec_rows = table_copy['subject_id'] == spec_id
        before_beh = table_copy[spec_rows]['beh_before']
        after_beh = table_copy[spec_rows]['beh_after']
        trans[spec_id] = [(b, a) for b, a in zip(before_beh, after_beh)]

    return trans, table_copy


def generate_standard_id_for_full_annots(annot_file: str):
    """ Gnerates a standard id for a file of full annotations.

    See documentation for generate_standard_id_for_volume for the format of a standard id.
    """
    match_str = '(?P<yr>.+)-(?P<mn>.+)-(?P<dy>.+)L(?P<sn>.+)-behavior.+'
    match = re.search(match_str, annot_file)
    # Note the sn key will contain any extra information by the way we broke up the reg expression
    return 'CW_' + match['yr'] + '-' + match['mn'] + '-' + match['dy'] + '-L' + match['sn']


def generate_standard_id_from_matlab_id(id: str) -> str:
    """ Generates a specimen ID in the format of the original excel files from an id in a matlab file.

    This function used to be called generate_excel_id_from_matlab_id.

    See documentation for generate_standard_id_for_volume for the format of a standard id.
    """

    yr = '17'
    mn = id[0:2]
    dy = id[2:4]
    sn = id[4:6]

    # Handle special cases
    if id == '0824L2CL':
        extra = '-1'
    elif id == '0824L2-2CL':
        extra = '-2'
    else:
        extra = ''

    return 'CW_' + yr + '-' + mn + '-' + dy + '-' + sn + extra


def generate_standard_id_for_volume(main_folder: str, sub_folder:str):
    """ Generates a standard string for the id of a subject, given a path to it's registered imaging data.

    Standard form subject names are of the form CW_<yr>-<mn>-<dy>-L<n>[-extra], where [-extra] includes
    optional extra information.

    Args:
        main_folder: The main folder for the subject

        sub_folder: The subfolder for the subject

    Returns:

        subj_str: A string identifying the subject in standard form

    """
    match = re.search('(?P<subject>.+)-561nm.+', sub_folder)
    subject = match['subject']
    return main_folder + '-' + subject


def generate_transition_dff_table(act_data: dict, trans: dict,
                                  spec_id_var_name: str = 'newTransitions',
                                  before_var_name: str = 'activityPreManipulationSet',
                                  dur_var_name: str = 'activityDurManipulationSet',
                                  after_var_name: str = 'activityPostManipulationSet'):
    """ Generates a table of transitions with dff values from the second version of data extracted by Chen.

    This will produce a table of the same format as produce_table_of_extracted_data.  The difference between this
    function and that function is the format of input data they accept.

    Args:
        act_data: The data as loaded from the original MATLAB files with delta F/F values in them.

        spec_id_var_name: The name of the varialbe in the MATLAB file containing the id of each specimen.

        before_var_name, dur_var_name, after_var_name: The names of the variables in the MATLAB data containing neural
        activity before, during and after the manipulations.

        trans: A dictionary of transitions, as produced by the function extract_transitions

    Returns:
        table: Each row represents the activity recorded from one neuron for one event.  Columns are:

            subject_id: The subject identifier for the specimen the cell was recorded in

            cell_id: The number identifier for the cell

            event_id: The event id, which corresponds to the order of the event, starting with 0, in the experiment

            beh_before: The behavior before the event

            beh_after: The behavior after the event

            dff_before: The Delta F/F value before the event

            dff_during: The Delta F/F value during the event

            dff_after: The Delta F/F value after the event

    Raises:
        ValueError: If any nan values are found in the neural activity
    """

    act_spec_ids = act_data[spec_id_var_name]
    before_act = act_data[before_var_name]
    dur_act = act_data[dur_var_name]
    after_act = act_data[after_var_name]

    n_events = [act.shape[1] - 1 for act in before_act]

    # Run checks on neural activity
    act_list = [before_act, dur_act, after_act]
    for act in act_list:
        for s_i, s_act in enumerate(act):
            if np.any(np.isnan(s_act[:, 1:])):
                raise(ValueError('Caught nan values in activity data for subject ' + str(s_i) + '.'))

        for s_i, spec_act in enumerate(act):
            if spec_act.shape[1] - 1 != n_events[s_i]:
                raise(ValueError('Caught different number of events for same specimen in activity data.'))

    # Convert ids in the activity files (MATLAB) to the format they are in excel
    spec_ids = [generate_standard_id_from_matlab_id(id) for id in act_spec_ids]

    # =================================================================================================
    # Produce DataFrame here

    def _data_frame_for_spec(subject_id, subject_act, subject_trans):
        """ Helper function to form one table for a single specimen.

        Args:
            subject_id: The id for the subject

            subject_act: List of activity in order before, during, after

            spec_annots: List of transitions

        Returns:
            The table for the specimen

        """

        col_names = ['subject_id', 'cell_id', 'event_id', 'beh_before', 'beh_after',
                                         'dff_before', 'dff_during', 'dff_after']

        s_n_events = subject_act[0].shape[1] - 1

        s_n_neurons = subject_act[0].shape[0]

        #print('subject_id: ' + subject_id)
        #print('s_n_events: ' + str(s_n_events))
        #print('subject_trans: ' + str(subject_trans))

        # Generate table
        s_table = pd.DataFrame(columns=col_names)

        for n_i in range(s_n_neurons):
            for e_i in range(s_n_events):
                cur_row = {'event_id': e_i,
                           'subject_id': subject_id,
                           'cell_id': subject_act[0][n_i, 0],
                           'beh_before': subject_trans[e_i][0],
                           'beh_after': subject_trans[e_i][1],
                           'dff_before': subject_act[0][n_i, e_i + 1],
                           'dff_during': subject_act[1][n_i, e_i + 1],
                           'dff_after': subject_act[2][n_i, e_i + 1]}
                s_table = s_table.append(cur_row, ignore_index=True)

        return s_table

    full_data_frame = pd.DataFrame()
    for s_i, spec_id in enumerate(spec_ids):
        spec_act = [before_act[s_i], dur_act[s_i], after_act[s_i]]
        full_data_frame = full_data_frame.append(_data_frame_for_spec(spec_id, spec_act, trans[spec_id]),
                                                 ignore_index=True)

    return full_data_frame


def generate_roi_dataset(img_folder: pathlib.Path, img_ext: str, frame_rate: float,
                         roi_dicts: Sequence[dict], metadata: dict, run_checks: bool = True,
                         add_images: bool = True):
    """ Generates a dataset of ROIS extracted from whole brain videos.

    Args:

        img_folder: The folder containing image folders.  Each image folder should contain one
        raw image for a time point.

        img_ext: The extension of image files.  This will be used to select the correct image
        file from an image folder if there are multiple image files present in each folder.

        frame_rate: The frame rate images were processed at

        roi_dicts: roi_dicts[i] contains a dictionary with information about the i^th set of extracted rois.
        Each dictionary should contain the keys:

            'group_name': The name to use for the group of rois

            'roi_locs_file': The path to the .pkl file with the roi locations in it

            'roi_values': A list.  Each entry in the list corresponds to a set of extracted values from
            the rois and is a dictionary with the keys 'file': with the path to the .h5 file with the extracted
            values and 'name' with the name that should be used in the dataset ts_data dict for these values

            'extra_attributes': A dictionary of extra attributes that should be saved with the group.  This
            key can be optionally omitted.

        metadata: A dictionary of metadata to save with the dataset

        run_checks: If true, checks for consistency between the different sources of data that make up the dataset

    Raises:

        RuntimeError: If the number of time points in any of the extracted ROI values does not match the number
        of images in the dataset.

        RuntimeError: If the number of ROIs in any of the extracted ROI values does not match the number of ROIs
        in the corresponding ROI locations file.
    """

    if add_images:
        # Find our images
        image_names_sorted = find_images(pathlib.Path(img_folder), img_ext,
                                         image_folder_depth=1, verbose=True)

        # Convert from paths to strings - this is to ensure compatability across operating systems. Also put
        # images paths in dictionaries - this will allow us to add additional fields to store with each
        # image name later
        image_names_sorted = [{'file': str(i_name)} for i_name in image_names_sorted]

        # Put images into a dictionary for time stamp data
        n_images = len(image_names_sorted)
        image_int = 1.0/frame_rate
        image_ts = np.asarray([image_int*i for i in range(n_images)])
        im_dict = {'ts': image_ts, 'vls': image_names_sorted}

        data_dict = {'imgs': im_dict}
    else:
        image_ts = np.asarray([0.0])
        data_dict = dict()

    # Now we process rois
    roi_groups = dict()
    for roi_dict in roi_dicts:

        # Read in ROI locations
        with open(roi_dict['roi_locs_file'], 'rb') as f:
            group_rois = pickle.load(f)

        group_rois = [ROI.from_dict(r) for r in group_rois]
        n_group_rois = len(group_rois)

        # Process extracted values for the rois
        for v_dict in roi_dict['roi_values']:
            values_folder, values_file = os.path.split(v_dict['file'])

            values = NDArrayHandler(folder=values_folder, file_name=values_file)

            if run_checks:
                # Make sure the values have the expected number of rois and time stamps
                if add_images:
                    n_vl_ts, n_vl_rois = values[:].shape
                    if n_vl_ts != n_images:
                        raise(RuntimeError('Dataset has ' + str(n_images) + ' images but found ' +
                                            str(n_vl_ts) + ' data points in ' + str(v_dict['file']) + '.'))
                if n_vl_rois != n_group_rois:
                    raise(RuntimeError('Group has ' + str(n_group_rois) + ' ROIS but found ' +
                                        str(n_vl_rois) + ' ROIS in ' + str(v_dict['file']) + '.'))

            # Add ROI values to data dictionary
            values_dict = {'ts': image_ts, 'vls': values}
            data_dict[v_dict['name']] = values_dict

        # Add the information for these rois to the roi_groups dict
        group_ts_labels = [d['name'] for d in roi_dict['roi_values']]

        if 'extra_attributes' not in roi_dict:
            extra_attributes = dict()
        else:
            extra_attributes = roi_dict['extra_attributes']

        roi_groups[roi_dict['group_name']] = {'rois': group_rois, 'ts_labels': group_ts_labels,
                                              **extra_attributes}
    # Create the dataset
    return ROIDataset(ts_data=data_dict, metadata=metadata, roi_groups=roi_groups)


def match_annotation_subject_to_volume_subject(vol_subject_main_folder: str, vol_subject_sub_folder: str,
                                               annot_subjects: Sequence[str]) -> int:
    """ Finds annotations for a set of registered images.

    Args:
        vol_subject_main_folder: The main folder containing the registered images

        vol_subject_sub_folder: The sub folder containing the registered images

        annot_subjects: List of subject ids we have annotations for.

    Returns:
        ind: The index into annot_subjects for the annotations corresponding to the registered images.  If no
        annotations exist, returns None

    """

    match = re.search('(?P<subject>.+)-561nm.+', vol_subject_sub_folder)
    subject = match['subject']
    annot_match_str = vol_subject_main_folder + '-' + subject

    # See if we can find this subject in the annotations
    trans_i = np.argwhere(annot_subjects == annot_match_str)
    if trans_i.size > 0:
        if trans_i.size > 1:
            raise(RuntimeError('Caught multiple matches'))
        else:
            return trans_i[0][0]
    else:
        return None


def match_standard_subject_ids(match_id: str, id_list:Sequence) -> int:
    """ Searches a list of subject ids and returns the index of a match.

    Args:
        match_id: The subject id to search for

        id_list: A sequence of ids to search through

    Returns:

        match_ind: The index of the match.  If no match is found, returns None.

    Raises:

        ValueError: If more than one match is found.

    """

    match_ind = None
    for i, cur_id in enumerate(id_list):
        if cur_id == match_id:
            if match_ind is None:
                match_ind = i
            else:
                raise(ValueError('Found duplicate matches.'))
    return match_ind

def produce_table_of_extracted_data(act_data: dict, annot_data: dict,
                                    before_var_name: str = 'activityPreManipulationSet',
                                    dur_var_name: str = 'activityDurManipulationSet',
                                    after_var_name: str = 'activityPostManipulationSet',
                                    annot_var_name: str = 'transitions') -> pd.DataFrame:
    """ Produces a DataFrame from extracted data originally saved in MATLAB format.  It is specifically for data
    formats provided by Chen Wang in the early parts of the VNC project.

    This function is specifically for processing data originally provided by Chen Wang, with the Delta F/F values
    extracted in windows before, during or after a manipulation event for different cells and across different
    specimens.  It will return a table represented in a Panda's DataFrame, where each row is the activity recorded
    from one neuron for one event.

    This function will run multiple checks of data integrity.  See list of exceptions that can be raised below for list
    of things that are checked.

    Args:
        act_data: A dictionary returned by scipy.load of the neural activity.

        annot_data: A dictionary returned by scipy.load of the annotations for each event.

        before_var_name, dur_var_name, after_var_name: The names of the variables in the MATLAB data containing neural
        activity before, during and after the manipulations.

        annot_var_name: The name of the variable in the MATLAB data containing the annotation data.

    Returns:
        table: Each row represents the activity recorded from one neuron for one event.  Columns are:

            subject_id: The subject identifier for the specimen the cell was recorded in

            cell_id: The number identifier for the cell

            event_id: The event id, which corresponds to the order of the event, starting with 0, in the experiment

            beh_before: The behavior before the event

            beh_after: The behavior after the event

            dff_before: The Delta F/F value before the event

            dff_during: The Delta F/F value during the event

            dff_after: The Delta F/F value after the event

    Raises:
        ValueError: If there is a different number of specimens between the different types of activity.
        ValueError: If there are different numbers of events for the different types of activity.
        ValueError: If the different type of activities list neurons in different orders.
        ValueError: If there is a different number of specimens in the activity and annotation data.
        ValueError: If there are different numbers of events between the activity and annotation data.
        ValueError: If any of the activity values are Nan.

    """

    before_act = act_data[before_var_name]
    dur_act = act_data[dur_var_name]
    after_act = act_data[after_var_name]

    annots = annot_data[annot_var_name]

    # =================================================================================================
    # Make sure number of specimens and events in different types of data is as expected
    n_specimens = len(before_act)
    n_events = [s.shape[1] - 1 for s in before_act]

    # Run checks on neural activity
    act_list = [before_act, dur_act, after_act]
    for act in act_list:
        if len(act) != n_specimens:
            raise(ValueError('Different number of speciments caught in activity data.'))
        if [s.shape[1] - 1 for s in act] != n_events:
            raise(ValueError('Different number of events caught in activity data.'))
        if np.any([np.any(np.isnan(s)) for s in act]):
            raise(ValueError('Caught nan values in activity data.'))

    for b_act, d_act, a_act in zip(*act_list):
        if (np.any(b_act[:, 0] != d_act[:, 0])) or (np.any(b_act[:, 0] != a_act[:, 0])):
            raise(ValueError('Caught different neuron orders across neural activity data.'))

    # Run checks on annotations
    if len(annots) != n_specimens:
        raise(ValueError('Different number of specimens caught between neural data and annotations.'))

    if [len(s) - 1 for s in annots] != n_events:
        raise(ValueError('Caught different numbers of events between neural data and annotations.'))

    # =================================================================================================
    # Produce DataFrame here

    def _data_frame_for_spec(spec_act, spec_annots):
        """ Helper function to form one table for a single specimen.

        Args:
            spec_act: List of activity in order before, during, after

            spec_annots: Annotation data for specimen

        Returns:
            The table for the specimen

        """

        col_names = ['subject_id', 'cell_id', 'event_id', 'beh_before', 'beh_after',
                                         'dff_before', 'dff_during', 'dff_after']

        s_n_events = len(spec_annots) - 1
        s_beh = spec_annots[0:-1]

        # Pull out name of specimen
        subject_id = spec_annots[-1]

        s_n_neurons = spec_act[0].shape[0]

        # Generate table
        s_table = pd.DataFrame(columns=col_names)

        for n_i in range(s_n_neurons):
            for e_i in range(s_n_events):
                cur_row = {'event_id': e_i,
                           'subject_id': subject_id,
                           'cell_id': spec_act[0][n_i, 0],
                           'beh_before': s_beh[e_i][0],
                           'beh_after': s_beh[e_i][1],
                           'dff_before': spec_act[0][n_i, e_i+1],
                           'dff_during': spec_act[1][n_i, e_i+1],
                           'dff_after': spec_act[2][n_i, e_i+1]}
                s_table = s_table.append(cur_row, ignore_index=True)

        return s_table

    full_data_frame = pd.DataFrame()
    for s_i in range(n_specimens):
        spec_act = [before_act[s_i], dur_act[s_i], after_act[s_i]]
        spec_annot = annots[s_i]
        full_data_frame = full_data_frame.append(_data_frame_for_spec(spec_act, spec_annot), ignore_index=True)

    return full_data_frame


def read_full_annotations(file: pathlib.Path) -> pd.DataFrame:
    """ Reads full annotations from a csv file.

    The full annotations are those provided by Nadine marking all events for a sample.

    This function will also return these annotations in a new format, matching the convention
    used in the rest of the project.

    Args:
        file: The csv file to open

    Returns:

        annots: The annotations.  This is a DataFrame with three columns: Start (marking
        the start of the annotation), End (marking the end of the annotation) and Beh (giving
        the annotation for the behavior. If the event was marked but did not correspond to a
        behavior (e.g., increased neural activity or something else of potential interest)
        it will have a value of np.nan. Note: Start and End frames are zero-indexed.

    Raises:

        RuntimeError: If a marked turn does not also include a marked direction.
    """

    BASIC_BEHS = ['F', 'B', 'S', 'H', 'T', 'O', 'P']
    TURN_BEHS = ['TL', 'TR']

    # Read in the original annotations
    annots = pd.read_csv(file, delimiter=';')

    # Rename the columns of the annotations to match our convention
    col_mapper = {k: dict_find(FULL_ANNOT_BEH_CODES, k) for k in annots.columns}
    col_mapper['START'] = 'Start'
    col_mapper['END'] = 'End'
    annots.rename(columns=col_mapper, inplace=True)

    # Get the annotated behavior for each event
    beh_col_dict = dict()
    for r in annots.index:
        event_annots = annots.loc[r, BASIC_BEHS].to_numpy()

        match_ind = np.argwhere(np.logical_not(np.isnan(event_annots)))
        if not match_ind.size == 0:
            beh = BASIC_BEHS[match_ind[0][0]]
            if beh == 'T':
                turn_annots = annots.loc[r, TURN_BEHS].to_numpy()
                turn_match_ind = np.argwhere(np.logical_not(np.isnan(turn_annots)))
                if turn_match_ind.size == 0:
                    raise (RuntimeError('Found a turn without a marked direction.'))
                beh = TURN_BEHS[turn_match_ind[0][0]]
            beh_col_dict[r] = beh
        else:
            beh_col_dict[r] = np.nan

    # Create a new formatted annotations data frame
    formatted_annots = annots[['Start', 'End']].copy()
    formatted_annots['Beh'] = pd.Series(beh_col_dict)

    # Adjust annotations so they are zero-indexed
    formatted_annots[['Start', 'End']] = formatted_annots[['Start', 'End']] - 1

    return formatted_annots


def read_raw_transitions_from_excel(file: Union[pathlib.Path, str], sheet_name: str='For Will',
                                smp_id_col: str = 'Date and sample',
                                prec_beh_col: str = 'Precede Behavior',
                                suc_beh_col: str = 'Succeed Behavior',
                                tgt_site_col: str = 'target site',
                                trans_time_col: str = 'transition time',
                                int_time_col: str = 'interval time',
                                manip_start_col: str = 'Manipulation start',
                                manip_end_col: str = 'Manipulation End',
                                adjust_frame_index: bool = True) -> pd.DataFrame:
    """ Reads in transitions for specimens from an excel spreadsheet with data in the final format for the project.

    Args:
        file: The excel file to extract transitions from

        sheet_name: The name of the sheet in the excel file with the transition information

        smp_id_col: The name of the column containing sample ids

        prec_beh_col: The name of the column labeling the preceeding behavior

        suc_beh_col: The name of the column labeling the succeeding behavior

        tgt_site_col: The name of the column giving the site that was targeted for perturbation

        trans_time_col: The name of the column giving the transition time

        int_time_col: The name of the column giving the interval time

        manip_start_col: The name of the column giving the index for manipulation start frames

        manip_end_col: The name of the column giving the index for manipulation end frames

        adjust_frame_index: True if indexing of start and end frames for the manipulation should be
        changed to 0 indexed (they are 1 indexed in the excel file).

    Returns:
        table: The data from the excel file, reformated
    """

    # Read in entire sheet
    df = pd.read_excel(file, sheet_name=sheet_name)

    # Down select to just the information we want
    df = df[[smp_id_col, prec_beh_col, suc_beh_col, tgt_site_col, trans_time_col, int_time_col,
             manip_start_col, manip_end_col]]

    # Rename our columns
    df.columns = ['subject_id', 'beh_before', 'beh_after', 'Tgt Site', 'Trans Time', 'Int Time',
                  'Manipulation Start', 'Manipulation End']

    if adjust_frame_index:
        df['Manipulation Start'] = df['Manipulation Start'] - 1
        df['Manipulation End'] = df['Manipulation End'] - 1

    return df


def recode_beh(table: pd.DataFrame, col):
    """ Recodes the behavioral labels in a Pandas Dataframe for consistency.

    Args:

        table: The table with data to recode

        col: The name of the column with behavioral labels for recoding

    Returns:
        new_table: The new table with the behavior recoded

    Raises:
        ValueError: If not all behaviors can be assigned a label
        ValueError: If one or more behaviors are assigned two new labels

    """
    # Make a copy of the table, so we return a new table
    table_copy = copy.deepcopy(table)
    orig_beh = table[col]

    # Define a boolean array to make sure we reassign each label exactly once
    previously_assigned = np.zeros(len(orig_beh), dtype=np.bool)

    # Perform recoding here; making sure we don't label anything twice
    for new_code, old_codes in BEHAVIOR_CODES.items():
        for old_code in old_codes:
            match_inds = orig_beh == old_code
            table_copy.loc[match_inds, col] = new_code

            match_inds_np = match_inds.to_numpy()
            if np.any(previously_assigned[match_inds_np]):
                raise(ValueError('Caught double label.'))
            previously_assigned[match_inds_np] = True

    # Make sure we relabled everything at least once
    if not np.all(previously_assigned):
        raise(ValueError('Unable to recogonize all existing labels.'))

    return table_copy
    pass


def count_transitions(table: pd.DataFrame, behs: Sequence[str] = None):
    """ Generates a table with the number of transitions.

    Args:

        table: The table of data, as produced by generate_transition_dff_table.

        behs: List of behaviors to look for transitions between.  If None, all behaviors in the table will be
        considered.

    Returns:
        table: The table with counts for each transition.  Rows are before behavior; columns are after behavior.
    """

    BEFORE_STR = 'beh_before'
    AFTER_STR = 'beh_after'

    if behs is None:
        behs = list(set(table[BEFORE_STR].unique().tolist() + table[AFTER_STR].unique().tolist()))
        behs.sort()

    n_behs = len(behs)
    n_trans = np.zeros([n_behs, n_behs])
    for b_i, b_b in enumerate(behs):
        for a_i, a_b in enumerate(behs):
            trans_rows = np.logical_and((table[BEFORE_STR] == b_b).to_numpy(),
                                        (table[AFTER_STR] == a_b).to_numpy())
            n_trans[b_i, a_i] = len(table[trans_rows])

    return pd.DataFrame(n_trans, index=behs, columns=behs)


def count_unique_subjs_per_transition(table: pd.DataFrame, behs: Sequence[str] = None):
    """ Generates a table with the number of subjects demonstrating a given transition.

    Args:
        table: The table of data, as produced by generate_transition_dff_table.

        behs: List of behaviors to look for transitions between. If None, all behaviors in the table
        will be considered.

    Returns:
        table: The table with counts of subjects for each transition.  Rows are before behavior; columns are
        after behavior.
    """

    BEFORE_STR = 'beh_before'
    AFTER_STR = 'beh_after'
    SMP_STR = 'subject_id'

    if behs is None:
        behs = list(set(table[BEFORE_STR].unique().tolist() + table[AFTER_STR].unique().tolist()))
        behs.sort()

    n_behs = len(behs)
    n_subjs_per_trans = np.zeros([n_behs, n_behs])
    for b_i, b_b in enumerate(behs):
        for a_i, a_b in enumerate(behs):
            trans_rows = np.logical_and((table[BEFORE_STR] == b_b).to_numpy(),
                                        (table[AFTER_STR] == a_b).to_numpy())
            n_subjs_per_trans[b_i, a_i] = len(table[trans_rows][SMP_STR].unique())

    return pd.DataFrame(n_subjs_per_trans, index=behs, columns=behs)

