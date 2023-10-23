import pickle
import numpy as np
import pandas as pd
import os
from cdasws import CdasWs
import datetime

cdas = CdasWs()
if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"


def process_goes_dataset(dataset):
    data = stack_from_data(dataset)
    data = fix_nan_for_goes(data)
    return data

def fix_nan_for_goes(data, nanvalue=-9998.0):
    data[data < nanvalue] = np.nan
    return data

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def stack_from_data(sc_data):
    bx, by, bz = sc_data[:, 0], sc_data[:, 1], sc_data[:, 2]
    b_gse_stacked = np.column_stack((bx, by, bz))
    return b_gse_stacked

def get_date_str_from_goesTime(goes_time):
    date_str = goes_time[0].strftime("%Y-%m-%d")
    return date_str

def stack_gk2a_data(sc_data):
    b_x = sc_data['b_xgse'][:]
    b_y = sc_data['b_ygse'][:]
    b_z = sc_data['b_zgse'][:]

    gk2a_bgse_stacked = np.column_stack((b_x, b_y, b_z))
    return gk2a_bgse_stacked

def fix_nan_for_goes(data, nanvalue=-9998.0):
    data[data < nanvalue] = np.nan
    return data

def goes_epoch_to_datetime(timestamp):
    """
    Converts goes epoch time from .cda into pandas datetime timestamp

    :param timestamp: from .cda file
    :return: pandas datetime timestamp
    """
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime

