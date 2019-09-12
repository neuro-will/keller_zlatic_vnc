""" Contains tools for reading in and converting data. """

import numpy as np
import pandas as pd

def produce_table_of_extracted_data(act_data: dict, annot_data: dict,
                                    before_var_name: str = 'activityPreManipulationSet',
                                    dur_var_name: str = 'activityDurManipulationSet',
                                    after_var_name: str = 'activityPostManipulationSet',
                                    annot_var_name: str = 'transitions') -> pd.DataFrame:
    """ Produces a DataFrame from extracted data original saved in MATLAB format.

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

            subject_id: The suject identifier for the specimen the cell was recorded in

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
        if (np.any(b_act[:,0] != d_act[:,0])) or (np.any(b_act[:,0] != a_act[:,0])):
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
            spec_act: List of activity in order during, before, after

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
                # Do some stuff
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






