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
                       enc_beh_interactions: bool = False, beh_interactions: list = None,
                       beh_before_str: str = 'beh_before', beh_after_str: str = 'beh_after'):
    """ Generates one-hot representation of data in tables produced by data_processing.produce_table_of_extracted data.

    Args:
        table: The table of data to process

        beh_before: A list of before behaviors to encode

        beh_after: A list of after behaviors to encode

        enc_subjects: True if subject id should be encoded

        enc_beh_interactions: True if all interaction terms between before and after behavior should be encoded.
        If provided, interactions_behs must be None.

        beh_interactions: A list of specific interaction terms of the form [('Q', 'F'), ...] to encode.  If provided,
        enc_beh_interactions must be false.

        beh_before_str: The column name in table that before behaviors are stored under
        beh_after_str: The column name in table that after behaviors are stored under

    Returns:

        encoding: The one hot encoded variables. Of shame n_smps*n_vars, where n_smps is the number of rows in table
        and n_vars is the number of encoded one-hot variables.  encoding[i,:] is the one hot encoding for the i^th row
        of table

        var_strs: String representation of each variable.  var_strs[j] is the name of the variable represented in the
        j^th column of encoding

    Raises:
        ValueError: If interaction_behs are provided and enc_beh_interactions is true.

    """

    if enc_beh_interactions is True and beh_interactions is not None:
        raise(ValueError('enc_beh_interactions must be false if beh_interactions are provided.'))

    n_smps = len(table)

    encoding = np.zeros([n_smps, 0])
    var_strs = []

    # Process before behaviors
    if beh_before is not None:
        n_before_beh = len(beh_before)
        beh_before_enc = np.zeros([n_smps, n_before_beh])
        for b_i in range(n_before_beh):
            beh_before_enc[:, b_i][table[beh_before_str] == beh_before[b_i]] = True
            var_strs.append(beh_before_str + '_' + beh_before[b_i])
        encoding = np.concatenate([encoding, beh_before_enc], axis=1)

    # Process after behaviors
    if beh_after is not None:
        n_after_beh = len(beh_after)
        beh_after_enc = np.zeros([n_smps, n_after_beh])
        for b_i in range(n_after_beh):
            beh_after_enc[:, b_i][table[beh_after_str] == beh_after[b_i]] = True
            var_strs.append(beh_after_str + '_' + beh_after[b_i])
        encoding = np.concatenate([encoding, beh_after_enc], axis=1)

    # Process all interaction terms if we are suppose to
    if enc_beh_interactions:
        n_before_beh = len(beh_before)
        n_after_beh = len(beh_after)

        beh_i_encoding = np.zeros([n_smps, n_before_beh*n_after_beh])

        i_col = 0
        for bb_i in range(n_before_beh):
            before_enc = np.zeros(n_smps)
            before_enc[table[beh_before_str] == beh_before[bb_i]] = True

            for ba_i in range(n_after_beh):
                after_enc = np.zeros(n_smps)
                after_enc[table[beh_after_str] == beh_after[ba_i]] = True
                beh_i_encoding[:, i_col] = before_enc*after_enc
                var_strs.append('beh_interact_' + beh_before[bb_i] + beh_after[ba_i])
                i_col+= 1

        encoding = np.concatenate([encoding, beh_i_encoding], axis=1)

    if beh_interactions is not None:
        n_interact_terms = len(beh_interactions)

        beh_i_encoding = np.zeros([n_smps, n_interact_terms])
        for bb_i in range(n_interact_terms):
            before_enc = np.zeros(n_smps)
            after_enc = np.zeros(n_smps)

            before_enc[table[beh_before_str] == beh_interactions[bb_i][0]] = True
            after_enc[table[beh_after_str] == beh_interactions[bb_i][1]] = True

            beh_i_encoding[:, bb_i] = before_enc*after_enc
            var_strs.append('beh_interact_' + beh_interactions[bb_i][0] + beh_interactions[bb_i][1])

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


# TODO: Remove this function, as it is no longer needed
def spont_beh_one_hot_encoding(tbl: pd.DataFrame, prev_str: str = None, suc_str: str = None, prev_ref: str = 'Q'):
    """ Produces one-hot encoding of previous behaviors and behaviors transitioned into.

    The previous behaviors will be referenced to a user specified behavior.

    Args:
        tbl: The table of events to encode.

        prev_str: The name of the column marking previous behaviors.

        suc_str: The name of the column marking behaviors transitioned into.

        prev_ref: The behavior to reference previous behaviors to.

    Returns:

        table: A numpy array of size n_events by n_variables, with a one hot encoding of previous behaviors and
        behaviors transitioned into.

        vars: A list of variable names, indicating which each column in table represents.
    """

    # Get list of before and after behaviors we will mark, taking into account the reference
    before_behs = np.asarray(list(set(list(tbl[prev_str].unique())) - set([prev_ref])))
    after_behs =  np.asarray(tbl[suc_str].unique())

    all_vars = ['before_' + b for b in before_behs] + ['after_' + b for b in after_behs]
    n_before_behs = len(before_behs)
    n_vars = len(all_vars)

    n_events = tbl.shape[0]
    one_hot = np.zeros([n_events, n_vars])
    for ev_i in range(n_events):
        before_beh_i = tbl[prev_str].iloc[ev_i]
        after_beh_i = tbl[suc_str].iloc[ev_i]
        before_match = np.argwhere(before_behs == before_beh_i)
        if len(before_match) > 0:
            one_hot[ev_i, before_match[0][0]] = 1
        after_match = np.argwhere(after_behs == after_beh_i)
        one_hot[ev_i, n_before_behs + after_match[0][0]] = 1

    return one_hot, all_vars


def reference_one_hot_to_beh(one_hot_data: np.ndarray, one_hot_vars: Sequence[str], beh: str,
                             remove_interaction_term: bool = True):
    """ Given a one hot encoding of behavioral variables, returns a one-hot encoding referenced to a given behavior.

    Args:

        one_hot_data: Array of one hot data of shape n_smps*n_vars, as returned by one_hot_from_table

        one_hot_vars: one_hot_vars[i] is a string with the name of the variable represented in the i^th column of
                      one_hot_data.

        beh: The behavior to reference to, e.g., 'Q'

    Returns:

        one_hot_data_ref: A one-hot representation of the data, referenced to the requested behavior

        one_hot_vars_ref: The variable names of the columns in one_hot_data_ref
    """

    before_match = np.argwhere([var == 'beh_before_' + beh for var in one_hot_vars])
    if len(before_match) > 0:
        before_ind = [before_match[0][0]]
    else:
        before_ind = []

    after_match = np.argwhere([var == 'beh_after_' + beh for var in one_hot_vars])
    if len(after_match) > 0:
        after_ind = [after_match[0][0]]
    else:
        after_ind = []

    if remove_interaction_term:
        interact_match = np.argwhere([var == 'beh_interact_' + beh + beh for var in one_hot_vars])
        if len(interact_match) > 0:
            interact_ind = [interact_match[0][0]]
        else:
            interact_ind = []
        print('interact_ind: ' + str(interact_ind))
    else:
        interact_ind = []

    del_inds = before_ind + after_ind + interact_ind

    print('del_inds: ' + str(del_inds))

    one_hot_data_ref = np.delete(one_hot_data, del_inds, axis=1)
    one_hot_vars_ref = [one_hot_vars[i] for i in range(len(one_hot_vars))
                        if i not in set(del_inds)]

    return [one_hot_data_ref, one_hot_vars_ref]


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


def order_and_color_interaction_terms(terms: Sequence[str], colors: dict = None, sort_by_before: bool = True):
    """ Produces an ordering that sorts interaction terms by behavior before or after manipulation and a color key.

    Colors are assigned to the terms so all terms starting (or optionally finishing) with the same behavior are
    assigned the same color.

    Args:
        terms: The interaction terms - two letter strings with the first letter indicating the behavior before the
        the manipulation and the second indicating the behavior after the manipulation.

        colors: A dictionary of colors to associate with each behavior.  Keys are letters indicating behaviors and
        values are rgb values.

        sort_by_before: True if we should sort interaction terms by the behavior that came before the manipulation,
        and this should be false if we should sort by the behavior that came after.
    """

    if sort_by_before:
        b_i = 0
    else:
        b_i = 1

    # Get sort order
    sort_behs = [b[b_i] for b in terms]
    order = np.argsort(sort_behs)

    # Get colors
    n_terms = len(terms)
    sorted_terms = [terms[i] for i in order]
    clrs = np.zeros([n_terms, 3])
    for t_i in range(n_terms):
        clrs[t_i, :] = colors[sorted_terms[t_i][b_i]]

    return [order, clrs]

