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
    capable of dividing total, complete and incomplete times in three different pandas.DataFrame.
    :param path: path to the .csv file
    :return: Dataframes
    """

    raw_data = pd.read_csv(path, parse_dates=['datetime'])
    rows_machine, columns_component = col_row_identification(raw_data)

    earliest_time = raw_data['datetime'].min()
    latest_time = raw_data['datetime'].max()

    dates, failures = date_maint_lists(raw_data, earliest_time, latest_time, rows_machine, columns_component)

    total_difs, incomp_difs, comp_difs = calculate_time_difs(dates, failures)

    total_times_frame = pd.DataFrame(data=np.array(total_difs), columns=columns_component, index=rows_machine)
    incomp_times_frame = pd.DataFrame(data=np.array(incomp_difs), columns=columns_component, index=rows_machine)
    comp_times_frame = pd.DataFrame(data=np.array(comp_difs), columns=columns_component, index=rows_machine)

    return total_times_frame, incomp_times_frame, comp_times_frame

# ----------------------------------- #
#       col_row_identification        #
# ----------------------------------- #


def col_row_identification(raw_data_frame):
    r"""
    For a given Dataframe, returns the unique values of the wanted rows and columns
    :param raw_data_frame: 
    :return: rows and columns names
    """

    rows = np.unique(raw_data_frame['machineID'].values)
    columns = np.unique(raw_data_frame['comp'].values)

    return rows, columns

# ----------------------------------- #
#      Dates and Corr/Pred lists      #
# ----------------------------------- #


def date_maint_lists(raw_data_frame, start_time, end_time, machines, components):
    r"""
    For the data frame, removes in list form the dates and type of work order, retaining machine ID and component
    :param raw_data_frame: Initial read data frame
    :param start_time: first registry of time in date column
    :param end_time: last registry of time in date column
    :param machines: list of machine ids
    :param components: list of component ids
    :return: two lists (dates and work orders types)
    """

    times_list = list()
    fails_list = list()
    for machine in machines:
        machine_fail_list = list()
        machine_list = list()

        for comp in components:
            aux_list = [start_time] + \
                       list(raw_data_frame[raw_data_frame['machineID'] == machine]
                            [raw_data_frame['comp'] == comp]['datetime']) + \
                       [end_time]
            aux_fail_list = list(raw_data_frame[raw_data_frame['machineID'] == machine]
                                 [raw_data_frame['comp'] == comp]['IF_FAIL']) + [0]

            machine_list.append(aux_list)
            machine_fail_list.append(aux_fail_list)

        times_list.append(machine_list)
        fails_list.append(machine_fail_list)

    return times_list, fails_list

# ----------------------------------- #
#      Dates and Corr/Pred lists      #
# ----------------------------------- #


def calculate_time_difs(times_list, fails_list):
    r"""
    Calculates the time between two maintenance operations and divides it or not in three lists
    :param times_list: list of maintenance operations dates
    :param fails_list: list of maintenance operations type
    :return: three lists, consisting of times between maintenance with indiscriminate type, corrective and preventive
    """
    total_times_list = list()
    incomp_times_list = list()
    comp_times_list = list()

    for i, machine in enumerate(times_list):
        total_machine_list = list()
        incomp_machine_list = list()
        comp_machine_list = list()

        for j, failure in enumerate(machine):
            total_failure_list = list()
            incomp_failure_list = list()
            comp_failure_list = list()

            for index in range(len(failure) - 1):
                total_failure_list.append((failure[index + 1] - failure[index]).days)

                if fails_list[i][j][index] == 0:
                    incomp_failure_list.append((failure[index + 1] - failure[index]).days)

                elif fails_list[i][j][index] == 1:
                    comp_failure_list.append((failure[index + 1] - failure[index]).days)

            total_machine_list.append(total_failure_list)
            incomp_machine_list.append(incomp_failure_list)
            comp_machine_list.append(comp_failure_list)

        total_times_list.append(total_machine_list)
        incomp_times_list.append(incomp_machine_list)
        comp_times_list.append(comp_machine_list)

    return total_times_list, incomp_times_list, comp_times_list
