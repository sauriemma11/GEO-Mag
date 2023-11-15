import pickle
import numpy as np
import pandas as pd
import os
from cdasws import CdasWs
import re
import datetime
import kp_data_processing as kp

cdas = CdasWs()
if "CDF_LIB" not in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"

def load_and_trim_data(pickle_dir, start_date, end_date):
    """
    Load and filter data from pickle files based on a specified date range.

    Args:
        pickle_dir (str): The directory containing pickle files with the data.
        start_date (datetime): The start date of the desired date range (
        inclusive).
        end_date (datetime): The end date of the desired date range (
        inclusive).

    Returns:
        tuple: A tuple containing four lists: (time_list, data_list,
        model_list, subtr_list).
            - time_list (list of datetime): List of datetime objects
            representing timestamps (sorted).
            - data_list (list): List of satellite position data points
            corresponding to each timestamp (sorted).
            - model_list (list): List of model data points corresponding to
            each timestamp (sorted).
            - subtr_list (list): List of subtracted data points
            corresponding to each timestamp (sorted).

    Note:
        This function loads data from pickle files in the specified
        directory, filters it
        based on the desired date range, and sorts all lists based on the
        time_list.
        The function assumes that the pickle files contain the following keys:
        'time_min', 'sat_gse', 'tsXX_gse', 'tsXX-sat', where 'XX' is a
        placeholder for the model identifier (e.g., '89' or '04').

    Example:
        To load and filter data from a directory 'data_dir' for the date
        range from 'start_date'
        to 'end_date', you can call the function as follows:

        start_date = datetime.datetime(2022, 4, 1)
        end_date = datetime.datetime(2022, 9, 30)
        time_list, data_list, model_list, subtr_list = load_and_trim_data(
        'data_dir', start_date, end_date)
    """
    time_list = []
    data_list = []
    model_list = []
    subtr_list = []

    for filename in os.listdir(pickle_dir):
        if filename.endswith('.pickle'):
            file_path = os.path.join(pickle_dir, filename)
            time, model_gse, sat_gse, subtr_data = \
                load_model_subtr_gse_from_pickle_file(
                file_path)

            # Convert time to datetime objects
            time = [pd.to_datetime(t) for t in time]

            # Filter data based on the desired date range
            filtered_time = []
            filtered_data = []
            filtered_model = []
            filtered_subtr = []

            for t, data, model, subtr in zip(time, sat_gse, model_gse,
                                             subtr_data):
                if start_date <= t <= end_date:
                    filtered_time.append(t)
                    filtered_data.append(data)
                    filtered_model.append(model)
                    filtered_subtr.append(subtr)

            # Extend the lists with filtered data
            time_list.extend(filtered_time)
            data_list.extend(filtered_data)
            model_list.extend(filtered_model)
            subtr_list.extend(filtered_subtr)

    # Sort all lists based on the time_list
    sorted_indices = sorted(range(len(time_list)), key=lambda i: time_list[i])
    time_list = [time_list[i] for i in sorted_indices]
    data_list = [data_list[i] for i in sorted_indices]
    model_list = [model_list[i] for i in sorted_indices]
    subtr_list = [subtr_list[i] for i in sorted_indices]

    return time_list, data_list, model_list, subtr_list


def load_subtr_data(file_path):
    """
    Load subtraction data from a pickle file.

    Args:
        file_path (str): The path to the pickle file containing the data.

    Returns:
        tuple: A tuple containing datetime values and subtraction data.
               - datetime_list (list): List of datetime values.
               - subtr_list (list): List of subtraction data.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        Exception: If there's an issue with loading the data from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # print(file)
            # Assuming data is a dictionary with keys 'datetime' and
            # 'subtraction'
            datetime_list = data.get('datetime', [])
            subtr_list = data.get('subtration', [])

            return datetime_list, subtr_list

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {file_path}")

    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {str(e)}")


def process_goes_dataset(dataset):
    """
    Process GOES dataset by stacking and fixing fill values

    :param dataset: 2D array of goes mag data (cart coords)
    :return: Processed dataset
    """
    data = stack_from_data(dataset)
    data = fix_nan_for_goes(data)
    return data


def fix_nan_for_goes(data, nanvalue=-9998.0):
    """
    Replaces fill values with nan (Default for GOES is -9999)

    :param data: dataset
    :param nanvalue:The value used to identify NaN values (default: -9998.0,
    which replaces -9999)
    :return: dataset with fill values replaced with nan
    """
    data[data < nanvalue] = np.nan
    return data


def load_model_subtr_gse_from_pickle_file(file_path):
    """
    Load a Python pickle file and return data for model and subtracted
    satellite data.

    The function dynamically determines the key for the model data by
    searching for a pattern.

    :param file_path: Path to the pickle file to load.
    :return: Tuple with time, model data, satellite position, and subtracted
    data.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        # print(f"KEYS: {data.keys()}")

        time = data.get('time_min', None)
        sat_gse = data.get('sat_gse', None)

        # Detect the pattern for the model data and construct the keys
        model_key_pattern = re.compile(r'ts\d+_gse')
        subtr_key_pattern = re.compile(r'ts\d+-sat')

        model_key = next(
            (key for key in data.keys() if model_key_pattern.match(key)), None)
        subtr_key = next(
            (key for key in data.keys() if subtr_key_pattern.match(key)), None)

        if model_key is None or subtr_key is None:
            raise KeyError("Model or subtracted satellite data key not found.")

        model_data = data.get(model_key)
        subtr_data = data.get(subtr_key)

        if model_data is None or subtr_data is None:
            raise ValueError("Model or subtracted satellite data not found.")

    return time, model_data, sat_gse, subtr_data

# sos_pickle = 'Z:/Data/GK2A/model_outputs/sosmag_modout_892019-04-01.pickle'
# sos_data = load_pickle_file(sos_pickle)
# print(sos_data.keys())
#
# g17_pickle = 'Z:/Data/GOES17/model_outs/G17_modout_892019-04-01.pickle'
# g17_data = load_model_gse_from_pickle_file(g17_pickle)
# print(g17_data.keys())

def stack_from_data(sc_data):
    """
    Stack data from the input array to make a 2D numpy array

    :param sc_data: array
    :return: Stacked np array
    """
    bx, by, bz = sc_data[:, 0], sc_data[:, 1], sc_data[:, 2]
    b_gse_stacked = np.column_stack((bx, by, bz))
    return b_gse_stacked


def get_date_str_from_goesTime(goes_time):
    """
    Get a date string in the format 'YYYY-MM-DD' or 'YYYY-MM-DD -> YYYY-MM-DD'
    from GOES timestamps, depending on the length of the time series.

    :param goes_time: GOES time object, assumed to be a list of datetime
    objects.
    :return: date string in the appropriate format.
    """
    if len(goes_time) == 1440:
        # One day's worth of data, return a single date string
        date_str = goes_time[0].strftime("%Y-%m-%d")
    elif len(goes_time) > 1441:
        # More than one day, return a range
        start_date_str = goes_time[0].strftime("%Y-%m-%d")
        end_date_str = goes_time[-1].strftime("%Y-%m-%d")
        date_str = f'{start_date_str} to {end_date_str}'
    else:
        # Handle unexpected case
        date_str = "Unknown range"
    return date_str


def stack_gk2a_data(sc_data):
    """
    Stack data from GK2A input array to make a 2D numpy stacked array
    ***Note the names of the array fields


    :param sc_data: Array, has fields 'b_xgse', 'b_ygse', and 'b_zgse'
    :return: stacked 2D numpy array
    """
    b_x = sc_data['b_xgse'][:]
    b_y = sc_data['b_ygse'][:]
    b_z = sc_data['b_zgse'][:]

    gk2a_bgse_stacked = np.column_stack((b_x, b_y, b_z))
    return gk2a_bgse_stacked


def goes_epoch_to_datetime(timestamp):
    """
    Converts goes epoch time from .cda into pandas datetime timestamp

    :param timestamp: from .cda file
    :return: pandas datetime timestamp
    """
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime
