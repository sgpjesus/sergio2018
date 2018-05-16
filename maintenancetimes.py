import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# ----------------------------------- #
#               Pipeline              #
# ----------------------------------- #


def main_pipeline(path):
    r"""
    For a given maintenance oriented .csv file, with the column scheme given by specification , this function is
    capable of dividing complete and incomplete times in two different pandas.DataFrame.
    :param path: path to the .csv file
    :return: Dataframes
    """

    raw_data = pd.read_csv(path, parse_dates=['datetime'])
    rows_machine, columns_component = col_row_identification(raw_data)

    earliest_time = raw_data['datetime'].min()
    latest_time = raw_data['datetime'].max()
    date_maint_lists()

    incomp_times_frame = pd.DataFrame(data=np.array(incomp_times_list), columns=columns_component, index=rows_machine)
    comp_times_frame = pd.DataFrame(data=np.array(comp_times_list), columns=columns_component, index=rows_machine)

    return incomp_times_frame, comp_times_frame

# ----------------------------------- #
#       col_row_identification        #
# ----------------------------------- #


def col_row_identification(raw_data_frame):
    r"""
    For a given Dataframe, returns the unique values of the wanted rows and columns
    :param raw_data_frame: 
    :return: rows and columns names
    """"

    rows = np.unique(raw_data_frame['machineID'].values)
    columns = np.unique(raw_data_frame['comp'].values)

    return rows, columns

# ----------------------------------- #
#      Dates and Corr/Pred lists      #
# ----------------------------------- #


def date_maint_lists():
    r"""
    For the dataframe, removes in list form the dates and type of work order, retaining machine ID and component
    :return: two lists (dates and work order)
    """

    times_list = list()
    fails_list = list()
    for machine in machineIDs:
        machine_fail_list = list()
        machine_list = list()
        for comp in comps:
            aux_list = [begin_time] + list(maint[maint['machineID'] == machine][maint['comp'] == comp]['datetime']) + [
                end_time]
            aux_fail_list = list(maint[maint['machineID'] == machine][maint['comp'] == comp]['IF_FAIL']) + [0]
            machine_list.append(aux_list)
            machine_fail_list.append(aux_fail_list)
        times_list.append(machine_list)
        fails_list.append(machine_fail_list)

    return times_list, fails_list

# ----------------------------------- #
#      Dates and Corr/Pred lists      #
# ----------------------------------- #


def calculate_time_difs():
    r"""

    :return:
    """

    incomp_times_list = list()
    comp_times_list = list()
    for i, machine in enumerate(times_list):
        incomp_machine_list = list()
        comp_machine_list = list()
        for j, failure in enumerate(machine):
            incomp_failure_list = list()
            comp_failure_list = list()
            for index in range(len(failure) - 1):
                if fails_list[i][j][index] == 0:
                    incomp_failure_list.append((failure[index + 1] - failure[index]).days)
                elif fails_list[i][j][index] == 1:
                    comp_failure_list.append((failure[index + 1] - failure[index]).days)
            incomp_machine_list.append(incomp_failure_list)
            comp_machine_list.append(comp_failure_list)
        incomp_times_list.append(incomp_machine_list)
        comp_times_list.append(comp_machine_list)

    return