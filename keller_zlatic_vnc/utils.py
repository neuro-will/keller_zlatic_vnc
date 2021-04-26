""" Contains basic utilities and helper functions.  """

import copy
import itertools
from typing import List


def form_combinations_from_dict(d: dict) -> List[dict]:
    """ Given a dictionary specifying combinations of parameters, forms individual dictionaries with all combinations.

    This function accepts as input a base dictionary, d.  It searches through the keys in the dictionary and identifies
    all keys with values which are lists.  Each of these is interpreted as containing a list of different values for
    the parameter with that key.  The function will then generate a list of dictionaries, where each has a single value
    for all keys and the full set of dictionaries contains all possible combinations of parameters.

    Example:

        d = {k_0: [0, 1], k_1: 1} would return the dictionaries {k_0: 0, k_1: 1} and {k_0: 1, k_1: 0}

    Args:

        d: The base dictionary.

    Returns:

        l: The list of dictionaries with all possible combinations of parameters.

    """

    # See which keys provide lists of values
    comb_keys = [k for k in d.keys() if isinstance(d[k], list)]
    comb_keys = [k for k in comb_keys if len(d[k]) > 1]

    if len(comb_keys) == 0:
        # Return the dictionary, making sure to strip the outer lists from any lists of length 1
        new_dict = {k: d[k] if not isinstance(d[k], list) else d[k][0] for k in d.keys()}
        return [new_dict]
    else:
        cur_key = comb_keys[0]
        cur_vls = d[cur_key]
        n_cur_vls = len(cur_vls)

        dict_list = [None]*n_cur_vls
        for vl_i, vl in enumerate(cur_vls):
            new_dict = copy.deepcopy(d)
            new_dict[cur_key] = vl
            dict_list[vl_i] = form_combinations_from_dict(new_dict)

        return list(itertools.chain(*dict_list))





