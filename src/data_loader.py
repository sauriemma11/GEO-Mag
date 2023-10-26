import pickle
import numpy as np
import pandas as pd
import os
from cdasws import CdasWs
import datetime

cdas = CdasWs()
if "CDF_LIB" not in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"


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


def load_pickle_file(file_path):
    """
    Load a python pickle file and return its data
    ***Currently unused

    :param file_path: path to pickle file to load
    :return: Loaded data from pickle file
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


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
    Get a date string in the format 'YYYY-MM-DD' from GOES timestamps. Used
    in plotting functions

    :param goes_time: GOES time object
    :return: date STR in YYYY-MM-DD format
    """
    date_str = goes_time[0].strftime("%Y-%m-%d")
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


# def fix_nan_for_goes(data, nanvalue=-9998.0):
#     data[data < nanvalue] = np.nan
#     return data

# ^ Repeat function? TODO: Delete if unneeded

def goes_epoch_to_datetime(timestamp):
    """
    Converts goes epoch time from .cda into pandas datetime timestamp

    :param timestamp: from .cda file
    :return: pandas datetime timestamp
    """
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime
