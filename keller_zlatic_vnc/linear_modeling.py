""" Tools for linear modeling work with Keller/Zlatic VNC data.

    William Bishop
    bishopw@hhmi.org
"""

import re
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def one_hot_from_table(table: pd.DataFrame, beh_before: list, beh_after: list, enc_subjects: bool = False,
                       enc_beh_interactions: bool = False):
    """ Generates one-hot representation of data in tables produced by data_processing.produce_table_of_extracted data.

    Args:
        table: The table of data to process

        beh_before: A list of before behaviors to encode

        beh_after: A list of after behaviors to encode

        enc_subjects: True if subject id should be encoded

        enc_beh_interactions: True if interaction terms between before and after behavior should be encoded

    Returns:

        encoding: The one hot encoded variables. Of shame n_smps*n_vars, where n_smps is the number of rows in table
        and n_vars is the number of encoded one-hot variables.  encoding[i,:] is the one hot encoding for the i^th row
        of table

        var_strs: String representation of each variable.  var_strs[j] is the name of the variable represented in the
        j^th column of encoding

    """

    n_smps = len(table)

    encoding = np.zeros([n_smps, 0])
    var_strs = []

    # Process before behaviors
    if beh_before is not None:
        n_before_beh = len(beh_before)
        beh_before_enc = np.zeros([n_smps, n_before_beh])
        for b_i in range(n_before_beh):
            beh_before_enc[:, b_i][table['beh_before'] == beh_before[b_i]] = True
            var_strs.append('beh_before_' + beh_before[b_i])
        encoding = np.concatenate([encoding, beh_before_enc], axis=1)

    # Process after behaviors
    if beh_after is not None:
        n_after_beh = len(beh_after)
        beh_after_enc = np.zeros([n_smps, n_after_beh])
        for b_i in range(n_after_beh):
            beh_after_enc[:, b_i][table['beh_after'] == beh_after[b_i]] = True
            var_strs.append('beh_after_' + beh_after[b_i])
        encoding = np.concatenate([encoding, beh_after_enc], axis=1)

    # Process interaction terms
    if enc_beh_interactions:
        n_before_beh = len(beh_before)
        n_after_beh = len(beh_after)

        beh_i_encoding = np.zeros([n_smps, n_before_beh*n_after_beh])

        i_col = 0
        for bb_i in range(n_before_beh):
            before_enc = np.zeros(n_smps)
            before_enc[table['beh_before'] == beh_before[bb_i]] = True

            for ba_i in range(n_after_beh):
                after_enc = np.zeros(n_smps)
                after_enc[table['beh_after'] == beh_after[ba_i]] = True
                beh_i_encoding[:, i_col] = before_enc*after_enc
                var_strs.append('beh_interact_' + beh_before[bb_i] + beh_after[ba_i])

                i_col+= 1

        encoding = np.concatenate([encoding, beh_i_encoding], axis=1)

    # Encode subjects
    if enc_subjects:
        unique_sub_ids = table['subject_id'].unique()
        n_subjects = len(unique_sub_ids)
        sub_id_enc = np.zeros([n_smps, n_subjects])
        for s_i, sub_id in enumerate(unique_sub_ids):
            sub_id_enc[:, s_i][table['subject_id'] == sub_id] = True
            var_strs.append('subject_' + sub_id)
        encoding = np.concatenate([encoding, sub_id_enc], axis=1)

    return [encoding, var_strs]


def color_grp_vars(var_strs: Sequence, colors: np.ndarray = None, c_map: str = 'tab10') -> list:
    """ Given a list of variables with prefixes indicating groups, this finds the groups and assigns each string a color
    based on it's group.

    Args:

        var_strs: The variable names of the format group_*.  If there is no underscore in a string, the entire the
        string will be taked as the string

        colors: The colors to assign to groups.  colors[i,:] will be assigned to the i^th group discovered in var_strs.
        If none, colors will be pulled from a color map.

        c_map: The color map to use when assigning colors if colors is None.  The alpha values of colors in a colormap
        will be ignored.

    Returns:
        clrs: A numpy array of shape n_strs*3.  clrs[i, :] is the RGB value for the color to use for var_strs[i]
    """

    n_strs = len(var_strs)

    # Get the group prefixes for each string
    re_str = re.compile(r".*_|.*")
    prefixes = [re_str.search(s).group(0) for s in var_strs]

    # See how many unique prefixes there are
    unique_prefixes = np.unique(prefixes)
    n_unique_prefixes = len(unique_prefixes)

    # Generate colors if we need to
    if colors is None:
        c_map_obj = plt.cm.get_cmap(c_map)
        colors = c_map_obj(np.arange(n_unique_prefixes))
        colors = colors[:, 0:3]

    # Return the dict
    clrs = np.zeros([n_strs, 3])
    for p_i, prefix in enumerate(prefixes):
        match_ind = unique_prefixes == prefix
        clrs[p_i, :] = colors[match_ind,:]

    return clrs


def format_whole_brain_annots_table(table: pd.DataFrame) -> pd.DataFrame:
    """ Formats a table of event annotations for use with the one_hot_from_table function.

    Formats a table produced by the script dff_extraction.ipynb.  The purpose is to change
    column labels and behavior annotations to make them consistent with one_hot_from_table.

    Args:
        table: The table to format

    Return:
        f_table: The formatted table

    """

    # Dictionary to convert column labels
    COL_ANNOT_DICT = {'Date and sample': 'subject_id',
                      'Precede Behavior': 'beh_before',
                      'Succeed Behavior ': 'beh_after'}

    # Dictionary to convert behavior annotations
    BEH_ANNOT_DICT = {'forward': 'F',
                  'quiet': 'Q',
                  'Quiet': 'Q',
                  'other': 'O',
                  'others': 'O',
                  'backward': 'B',
                  'turn': 'T',
                  'Forward': 'F',
                  'hunch': 'H',
                  'back hunch': 'BH',
                  'forw hunch': 'FH',
                  'forward hunch': 'FH'}

    # Rename columns
    table = table.rename(columns=COL_ANNOT_DICT)

    # Relabel behaviors
    n_events = len(table)
    before_check = np.zeros(n_events, dtype=np.bool)
    after_check = np.zeros(n_events, dtype=np.bool)

    for from_str, to_str in BEH_ANNOT_DICT.items():
        before_match  = table['beh_before'] == from_str
        after_match = table['beh_after'] == from_str
        table['beh_before'][before_match] = to_str
        table['beh_after'][after_match] = to_str
        before_check[before_match] = True
        after_check[after_match] = True

    if not np.all(before_check):
        raise(RuntimeError('Unable to relabel all before behaviors.'))
    if not np.all(after_check):
        raise(RuntimeError('Unable to relabel all after behaviors.'))

    return table


