import numpy as np
import datetime as dt
from typing import List, Tuple
import pandas as pd


def calculate_time_difference(longitude_degrees, hemisphere='W'):
    # Calculate the time difference for the input longitude
    # Input should be in degrees WEST. If east, (GK2A for example is at
    # 128.2 E) input degrees east + 360
    if hemisphere == 'E':
        longitude_degrees = 360 - longitude_degrees

    time_diff = (longitude_degrees / 360) * 24

    return time_diff


def unpack_components(data_list):
    # Unpack x, y, and z components from a list of data points
    x_component = [point[0] for point in data_list]
    y_component = [point[1] for point in data_list]
    z_component = [point[2] for point in data_list]
    return x_component, y_component, z_component


def calc_hourly_stddev(datetime_list, subtr_list, kp_mask=None):
    """
    Calculate hourly standard deviation of subtraction data with an optional
    Kp mask.

    Args:
        datetime_list (list): List of datetime values.
        subtr_list (list): List of subtraction data.
        kp_mask (list, optional): List of boolean values indicating Kp
        values over threshold.
                                  If provided, only data points where
                                  kp_mask is True will be used.

    Returns:
        pd.Series: Hourly standard deviation of subtraction data.
    """
    # Create a pandas DataFrame with datetime as the index
    df = pd.DataFrame({'datetime': datetime_list, 'subtraction': subtr_list})
    df.set_index('datetime', inplace=True)

    # Apply the Kp mask if provided
    if kp_mask is not None:
        df['subtraction'] = df['subtraction'].mask(kp_mask)

    # Resample the data to hourly frequency and calculate standard deviation
    hourly_std_dev = df['subtraction'].resample('H').std()

    return hourly_std_dev


def align_datasets(time_list_1: List, time_list_2: List,
                   data_1: List[float], data_2: List[float]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Align two datasets based on their timestamps and return the paired data.

    :param time_list_1: Timestamps of the first dataset.
    :param time_list_2: Timestamps of the second dataset.
    :param data_1: Data points of the first dataset.
    :param data_2: Data points of the second dataset.
    :return: Tuple of numpy arrays with aligned data from both datasets.
    """
    # Assuming the timestamps are already datetime objects and the lists are
    # sorted
    # Create a dictionary from time_list_2 to its corresponding data for
    # quick lookup
    data_dict_2 = {time: data for time, data in zip(time_list_2, data_2)}

    # Align the datasets
    aligned_data_1 = []
    aligned_data_2 = []
    for time, data in zip(time_list_1, data_1):
        if time in data_dict_2:
            aligned_data_1.append(data)
            aligned_data_2.append(data_dict_2[time])

    return np.array(aligned_data_1), np.array(aligned_data_2)

def calculate_std_dev(dataset1, dataset2):
    # Convert to arrays, handling potential raggedness
    array1 = np.array(dataset1, dtype=object) if isinstance(dataset1[0],
                                                            list) else \
        np.array(
            dataset1)
    array2 = np.array(dataset2, dtype=object) if isinstance(dataset2[0],
                                                            list) else \
        np.array(
        dataset2)

    # Calculate difference
    # Ensure both arrays are the same length and shape
    if array1.shape == array2.shape:
        difference = array1 - array2
        # Flatten the array to ensure np.nanstd can work on it
        difference = np.hstack(difference.ravel())
    else:
        raise ValueError("The inputs must have the same shape.")

    return np.nanstd(difference)


def calculate_total_magnetic_field(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def convert_timestamps_to_numeric(timestamps):
    # Convert timestamps to numeric values (e.g., seconds since a reference
    # date)
    numeric_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
    return numeric_values

def calc_line_of_best_fit(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(
        y)  # Ensure x and y are arrays and create the mask
    x_masked = x[mask]
    y_masked = y[mask]
    coeff = np.polyfit(x_masked, y_masked, 1)
    polynomial = np.poly1d(coeff)
    return polynomial


def get_avg_data_over_interval(time_list, data_list, interval='60T'):
    df = pd.DataFrame({'time': time_list, 'data': data_list})
    df.set_index('time', inplace=True)
    df_resampled = df.resample(interval).mean()

    return df_resampled.index.tolist(), df_resampled['data'].tolist()


def find_noon_and_midnight_time(time_diff, date_str, gk2a=False):
    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    date_obj_previous_day = date_obj - dt.timedelta(
        days=1)  # For plotting noon time GK2A

    midnight_time_UT = dt.datetime(date_obj.year, date_obj.month, date_obj.day,
                                   0, 0)
    noon_time_UT = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 12,
                               0)

    if gk2a is True:
        noon_time = dt.datetime(date_obj_previous_day.year,
                                date_obj_previous_day.month,
                                date_obj_previous_day.day, 12, 0)
        noon_time = noon_time + dt.timedelta(hours=time_diff)
        midnight_time = midnight_time_UT + dt.timedelta(hours=time_diff)
    else:
        midnight_time = midnight_time_UT + dt.timedelta(hours=time_diff)
        noon_time = noon_time_UT + dt.timedelta(hours=time_diff)

    return noon_time, midnight_time

# print(find_noon_and_midnight_time(calculate_time_difference(128.2),
# '2022-08-15', gk2a=True))
