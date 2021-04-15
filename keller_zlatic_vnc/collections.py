""" Contains tools for forming and working with collections. """

import copy
import datetime
import itertools
import os
from pathlib import Path
import pickle
import shutil
from typing import List, Sequence, Tuple, Union


import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.comments import CommentedSeq

# Define aliases
PathOrStr = Union[Path, str]


def form_collection(image_folder: PathOrStr, tgt_folder: PathOrStr,
                    responsible: List[str], description: str, git_hashes: dict,
                    preceding_behs: List[str], suceeding_behs: List[str],
                    params: Sequence[dict], ignore_extensions: Sequence[str]):
    """ Forms a collection of maps and associated files.

    This function is meant to be used to form a "basic" collection - a "basic" collection is made up of
    the maps produced by running a statistical pipeline with a single set of parameters once.  Collections
    can be combined (see the appropriate functions) after they are created, to create larger collections
    with maps for different parameter settings.

    A collection consists of the following files:

        1) Multiple maps.  These may consist of multiple different types of files.  There may be multiple files
        associated with one map (e.g., coefficients stored in one file, p-values in another file, and we might
        save the same data in different formats (.tiff files for viewing in FIJI as well as movies for quick
        reference).  Each name of the original files should start with the behavior or behavioral transitions
        the map is for (e.g., 'B_F_*.tiff' or 'B_*.tiff').  It is possible to include other files in the collection,
        such as pickle files storing colormaps - this can have any file name the user would like.  All files in
        the image_folder directory will be copied into the collection, unless they end with an extension that
        should be ignored (see the ignore_extensions argument below).

        2) A metadata.yaml file giving the value of important parameters in the collection.

        3) A 'parameters' folder containing saved .pkl files with the original parameter dictionaries used in
        the pipeline to produce the results.  These can be inspected for debugging purposes as well as to find
        the value of parmeters not listed in the metadata.yaml file.

    Args:

        image_folder: Path to folder containing the produced maps for the collection.

        tgt_folder: The folder that the collection should be created in.  This can be the existing image_folder; in
        this case, the other required files for the collection will be added to the tgt_folder.

        responsible: A list of those who are responsible for the collection

        description: A string describing the collection.

        git_hashes: Keys in this dictionary are the names of repositories.  Values give SHA-1 hashes of commits
        used to generate the results.

        preceding_behs: A list of preceding behaviors represented in the maps

        succeeding_behs: A list of succeeding behaviors represented in the maps

        params: A list of dictionaries with parameters we should add to the yaml file. Each
        entry should be a dictionary with the keys:

            'for_metadata' containing a dictionary with parameters for the yaml file.  Not all
            parameters in this dictionary need to be added to the yaml file (see inc_params).

            'inc_params' A dictionary with keys which give the keys in the for_yaml
            dictionary that should actually be saved in the yaml file and values which
            give a comment to save with the parameter.

        ignore_extensions: A list of file extensions to ignore when copying files in the image_folder
        to the collection.

    """

    image_folder = Path(image_folder)
    tgt_folder = Path(tgt_folder)

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # Generate the metadata file
    generate_metadata_file(file_path=tgt_folder / 'metadata.yml',
                           responsible=responsible,
                           description=description,
                           git_hashes=git_hashes,
                           preceding_behs=preceding_behs,
                           succeeding_behs=suceeding_behs,
                           params=params)

    # Save the parameters
    param_folder = tgt_folder / 'parameters'
    if not os.path.exists(param_folder):
        os.makedirs(param_folder)

    for param_d in params:
        with open(param_folder / (param_d['desc'] + '.pkl'), 'wb') as f:
            pickle.dump(param_d['for_saving'], f)

    # Copy over image files if needed
    candidate_files = os.listdir(image_folder)
    for file in candidate_files:
        _, ext = os.path.splitext(file)
        if ext not in ignore_extensions:
            tgt_file = tgt_folder / file
            if not os.path.exists(tgt_file):
                shutil.copyfile(image_folder / file, tgt_file)


def generate_metadata_file(file_path: Path, description: str, responsible: List[str], git_hashes: dict,
                           preceding_behs: Sequence[str], succeeding_behs: Sequence[str], params: Sequence[dict]):
    """  Generates a new metadata file.

    Args:

        file_path: The path to the file that will be created. If this file already exists, it will be overwritten.

        description: A string describing the collection.

        responsible: A list of those who are responsible for the collection

        git_hashes: A dictionary.  Keys are repository names and values are hashes indicating specific commits
        the results in the collection were generated for.

        preceding_behs: A list of preceding behaviors represented in the maps

        succeeding_behs: A list of succeeding behaviors represented in the maps

        params: A list of dictionaries with parameters we should add to the yaml file. Each
        entry should be a dictionary with the keys:

            'for_metadata' containing a dictionary with parameters for the yaml file.  Not all
            parameters in this dictionary need to be added to the yaml file (see inc_params).

            'inc_params' A dictionary with keys which give the keys in the for_yaml
            dictionary that should actually be saved in the yaml file and values which
            give a comment to save with the parameter.


    """

    with open(file_path, 'wb') as f:

        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False

        yaml.dump({'responsible': responsible}, f)
        yaml.dump({'description': description}, f)

        cur_time = datetime.datetime.now().isoformat()
        yaml.dump({'creation_time': cur_time}, f)

        yaml.dump({'git_hashes': git_hashes}, f)

        m = CommentedMap()
        m.insert(1, 'preceding_behaviors', preceding_behs,
                 comment='List of preceeding behaviors represented among the maps in the collection.')
        yaml.dump(m, f)

        m = CommentedMap()
        m.insert(1, 'suceeding_behaviors', succeeding_behs,
                 comment='List of suceeding behaviors represented among the maps in the collection.')
        yaml.dump(m, f)

        # Process the parameters
        yaml_param_d = CommentedMap()
        for p_i, param_d in enumerate(params):
            if len(param_d['inc_params']) > 0:
                keep_keys = [k for k in param_d['for_metadata'].keys()
                             if k in param_d['inc_params'].keys()]

                yaml_param_d_i = CommentedMap()
                for k in keep_keys:
                    yaml_param_d_i.insert(1, k, _seq_to_str(param_d['for_metadata'][k]),
                                          comment=param_d['inc_params'][k])
                yaml_param_d.insert(1, param_d['desc'], yaml_param_d_i)

        m = CommentedMap()
        m.insert(1, 'parameters', yaml_param_d)
        yaml.dump(m, f)


def merge_collections(ind_collections: Sequence[PathOrStr], tgt_folder: PathOrStr,
                      new_desc: str, ignore_keys: list=None, ignore_extensions: list = None):
    """ Merges individual collections into one.

    A merged collection is collection of files consisting of the following:

        1) A metadata file describing the contents and important parameters used in the creation of the
        maps in the merged collection.  When collections contain maps that were created with different
        parameters, the list of all values that were used across maps will be listed for the parameter
        in the metadata file.  Importantly, maps do not have to exists for all possible combinations of
        parameter values.

        2) A collection of images and movies showing the maps as well as associated other files (e.g., colormaps
        saved as pkl files).  When maps are created with different parameters, the parameter used to create a
        particular map will be noted in the filename by appending a string of <_parameter_name>_<parameter_value>
        to the filename.

        3) Folders containing .pkl files of the parameters used in the statistical pipeline that generated the data
        for the maps.  Each folder will also have strings of the form in (2) to indicate the particular combination
        of parameters the dictionaries in the folder pertain too.

        4) An original_collections.yaml file. This is for later reference, and indicates which individual collections
        were merged to generate the merged collection.

    Args:

        ind_collections: Paths to folders containing each individual collection

        tgt_folder: The folder the new collection should be created in.  If this folder does not
        exist, it will be created

        new_desc: A string providing a description for the merged collection.

        ignore_keys: Any keys from the metadata structures that we should ignore differences in for the
        purposes of naming files.  By default, if a there is a difference in a parameter values across
        metadata structures, when copying files from each individual collection to the merged collection
        these will be noted in the file names, as explained above.  However, some entries in a metadata
        structure, such as the description and creation time, should be ignored for this purpose, and the
        user can specify those keys here.  Note that the keys 'description', 'creation_time'
        'janelia_core' and 'keller_zlatic_vnc' will always be ignored.

        ignore_extensions: The extensions of any files that should not be included in the merged collection.

    """

    ALWAYS_IGNORE_KEYS = ['description', 'creation_time', 'janelia_core', 'keller_zlatic_vnc']

    if ignore_keys is not None:
        ignore_keys = ignore_keys + ALWAYS_IGNORE_KEYS
    else:
        ignore_keys = ALWAYS_IGNORE_KEYS

    if ignore_extensions is None:
        ignore_extensions = []

    # Make sure all inputs are Path objects
    ind_collections = [Path(coll_i) for coll_i in ind_collections]
    tgt_folder = Path(tgt_folder)

    # Load the metadata structures from each collection
    n_collections = len(ind_collections)
    ind_metadata = [None]*n_collections
    for i, coll_i in enumerate(ind_collections):
        with open(Path(coll_i) / 'metadata.yml', 'rb') as f:
            yaml=ruamel.yaml.YAML(typ='rt')
            ind_metadata[i] = yaml.load(f)

    merged_metadata, diff_keys = merge_metadata(ind_metadata)

    print(type(diff_keys[1]))

    # Overwrite the fields we need to in the merged metadata structure
    merged_metadata['description'] = new_desc

    cur_time = datetime.datetime.now().isoformat()
    merged_metadata['creation_time'] = cur_time

    # Create the folder, if needed, for the new collection
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    # Save the new merged metadata structure
    with open(tgt_folder / 'metadata.yaml', 'wb') as f:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        yaml.dump(merged_metadata, f)

    # Ignore keys with differences that we don't want to pay attention to
    diff_keys = [vl for vl in diff_keys if vl[0] not in ignore_keys]

    # Generate the strings we append to the file names for all collections
    coll_strs = [''.join(['_' + key + '_' + str(vls[c_i]) for (key, vls) in diff_keys]) for c_i in range(n_collections)]

    # Copy the files from each collection
    for coll_str, coll_i in zip(coll_strs, ind_collections):
        coll_files = os.listdir(coll_i)
        for src in coll_files:
            src_path = coll_i / src
            if (src_path.name != 'metadata.yaml') and (src_path.suffix not in ignore_extensions):
                tgt_path = tgt_folder / (src_path.stem + coll_str + src_path.suffix)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, tgt_path)
                else:
                    shutil.copy(src_path, tgt_path)

    # Generate the original collections metadata file
    orig_coll_yaml_path = tgt_folder / 'original_collections.yaml'
    with open(orig_coll_yaml_path, 'wb') as f:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = False
        yaml.dump({'original_collections': [str(coll_i) for coll_i in ind_collections]}, f)


def merge_metadata(m: Sequence[CommentedMap], _key=None) -> Tuple[CommentedMap, List[Tuple]]:
    """ Merges metadata structures from multiple collections.

    This function is designed for use when combining collections.  It will inspect the metadata structures of each
    individual collection.  It expects the metadata structures from each collection to have exactly the same
    structure and keys.   When keys across metadata files all have the same value, the merged metadata
    structure will take on that value for that key.  When keys have different values, the merged metadata
    structure for that key will be given the set of all key values found in the individual metadata
    structures.  In addition, this function will return the keys for which multiple values were found and indicate
    the inidividual values that were found for each metadata structure.

    Note that this function will preserve comments associated with the first metadata structure when merging.

    Args:
        m: m[i] is the metadata structure for the i^th collection.

        _key: An input that the user should not use.  This is provided for internal recursion by the function.

    Returns:

        merged_m: The merged metadata structure.

        diff_keys: Each entry in this list corresponds to keys for which we found different values across
        metadata structures.  Each entry will be a tuple of the form (key, values) where key is the
        key for which different values were found and values is a list of values, where values[i] is the
        value from the i^th metadata structure.

    """

    m_0 = m[0]

    if isinstance(m_0, CommentedMap):

        for m_i in m:
            if m_i.keys() != m_0.keys():
                raise(RuntimeError('Only metadata structures with the same fields can be merged.'))

        # We copy one of the commented maps as an easy way of keeping comments intact
        m_new = copy.deepcopy(m_0)

        n_keys = len(m_0.keys())
        diff_keys = [None]*n_keys
        for k_i, key in enumerate(m_0.keys()):
            new_entry, diff_keys[k_i] = merge_metadata([m_i[key] for m_i in m], _key=key)
            m_new[key] = new_entry

        # Merge are lists of keys with differences
        diff_keys = [r for r in diff_keys if r is not None]
        diff_keys = list(itertools.chain(*diff_keys))
        return m_new, diff_keys

    if isinstance(m_0, CommentedSeq):
        # Check if sequences contain all the same things
        same = True
        for m_i in m:
            if set(m_0) != set(m_i):
                same = False

        # If not, we merge them keeping track of the unique values from each sequence
        if same:
            return m_0, None
        else:
            m_new = copy.deepcopy(m_0)
            m_new.clear()
            all_values = [list(m_i) for m_i in m]
            merged_values = list(set(itertools.chain(*all_values)))
            merged_values.sort()
            for l_i in merged_values:
                m_new.append(l_i)
            return m_new, [(_key, all_values)]

    else:
        # Check if all the values are the same
        same = True
        for m_i in m:
            if m_i != m_0:
                same = False

        if same:
            return m_0, None
        else:
            key_record = [(_key, m)]
            merged_values = list(set(m))
            merged_values.sort()

            return merged_values, key_record


def _seq_to_str(sq):
    if isinstance(sq, Sequence) and not isinstance(sq, str):
        s = ''
        for v in sq:
            s += str(v) + '_'
        s = s[0:-1]
        return s
    else:
        return sq
