import numpy as np
import datetime as dt
from typing import List, Tuple
import pandas as pd
import re


def calculate_time_difference(longitude_degrees, hemisphere='W'):
    """
    Calculate the time difference based on a given longitude (degrees).

    Parameters
    ----------
    longitude_degrees (float): The longitude in degrees. Positive for east
    and negative for west.
    hemisphere (str, optional): The hemisphere ('E' for east, 'W' for west.
    Defaults to 'W'.

    Returns
    -------
    time_diff (float): The time difference in hours from the Greenwich Mean
    Time (GMT).
    """

    DEGREES_IN_CIRCLE = 360
    HOURS_IN_DAY = 24
    # Input should be in degrees WEST. If east, (GK2A for example is at
    # 128.2 E) input degrees east + 360
    if hemisphere == 'E':
        longitude_degrees = DEGREES_IN_CIRCLE - longitude_degrees

    time_diff = (longitude_degrees / DEGREES_IN_CIRCLE) * HOURS_IN_DAY

    return time_diff


def unpack_components(data_list):
    """
    Unpack xyz components from a list of data points.

    Parameters
    ----------
    data_list (List[Tuple[float, float, float]]): A list of tuples,
    where each tuple contain x, y, z components.

    Returns
    -------
    Tuple[List[float], List[float], List[float]]: Separate lists of x, y,
    and z components.

    """
    x_component, y_component, z_component = zip(*data_list)
    return list(x_component), list(y_component), list(z_component)

def calc_hourly_stddev(datetime_list, subtr_list, kp_mask=None):
    """
    Calculate hourly standard deviation of subtraction data with an optional
    Kp mask.

    Parameters
    ----------
    datetime_list (list): List of datetime values.
    subtr_list (list): List of subtraction data.
    kp_mask (list, optional): List of boolean values indicating Kp values
    over threshold. Default is None.
        If provided, only data points where kp_mask is True will be used.

    Returns
    -------
    hourly_std_dev (pd.Series): Hourly standard deviation of subtraction data.

    """
    # Create a pandas DataFrame with datetime as the index
    df = pd.DataFrame({'datetime': datetime_list, 'subtraction': subtr_list})
    df.set_index('datetime', inplace=True)

    # apply the Kp mask
    if kp_mask is not None:
        df['subtraction'] = df['subtraction'].mask(kp_mask)

    # resample data to hourly freq, take std dev
    hourly_std_dev = df['subtraction'].resample('H').std()

    return hourly_std_dev


def align_datasets(time_list_1: List, time_list_2: List,
                   data_1: List[float], data_2: List[float]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Align two datasets based on their timestamps and return the paired data.

    Parameters
    ----------
    time_list_1 : List[dt.datetime]
        Timestamps of the first dataset.
    time_list_2 : List[dt.datetime]
        Timestamps of the second dataset.
    data_1 : List[float]
        Data points of the first dataset.
    data_2 : List[float]
        Data points of the second dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two numpy arrays with aligned data from both datasets.
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
    """
    Calculate the standard deviation of the differences between two datasets.

    Parameters
    ----------
    dataset1 (List[float]): The first dataset.
    dataset2 (List[float]): The second dataset.

    Returns
    -------
    float: The standard deviation of the differences between the two datasets.

    Raises
    ------
    ValueError: If the input datasets do not have the same shape.
    """
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
    """
    Calculate the total magnetic field from its x, y, and z components.

    Parameters
    ----------
    x (np.ndarray): The x component of the magnetic field.
    y (np.ndarray): The y component of the magnetic field.
    z (np.ndarray): The z component of the magnetic field.

    Returns
    -------
    total_field (np.ndarray): The total magnetic field calculated from the
    x, y, and z components.
    """
    total_field = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return total_field


def convert_timestamps_to_numeric(timestamps):
    """
    Convert a list of datetime objects to numeric values representing
    seconds since a reference date.

    Parameters
    ----------
    timestamps (List[dt.datetime]): A list of datetime objects.

    Returns
    -------
    numeric_values (List[float]): Numeric values representing seconds since
    the first timestamp in the list.

    Raises
    ------
    TypeError: If any item in 'timestamps' is not a datetime object.
    ValueError: If the 'timestamps' list is empty.
    """
    if not all(isinstance(t, dt.datetime) for t in timestamps):
        raise TypeError("All items in 'timestamps' must be datetime objects.")
    if not timestamps:
        raise ValueError("The 'timestamps' list cannot be empty.")

    numeric_values = [(t - timestamps[0]).total_seconds() for t in timestamps]
    return numeric_values


def calc_line_of_best_fit(x, y):
    """
    Calculate the line of best fit for a set of x and y values.

    Parameters
    ----------
    x (List[float]): The x-values of the data points.
    y (List[float]): The y-values of the data points.

    Returns
    -------
    polynomial (np.poly1d): A polynomial representing the line of best fit.

    Raises
    ------
    TypeError: If 'x' or 'y' is not a list.
    ValueError: If 'x' and 'y' lists are not of equal length.

    """
    if not (isinstance(x, list) and isinstance(y, list)):
        raise TypeError("Both 'x' and 'y' should be lists.")
    if len(x) != len(y):
        raise ValueError("Length of 'x' and 'y' lists must be equal.")

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
    """
    Calculate the average data over a specified time interval.

    Parameters
    ----------
    time_list (List[dt.datetime]): List of datetime objects.
    data_list (List[float]): Corresponding data values.
    interval (str): Time interval for averaging, in pandas offset string
    format.

    Returns
    -------
    avg_data (Tuple[List[dt.datetime], List[float]]): The resampled time
    list and the corresponding averaged data.

    Raises
    ------
    TypeError: If 'time_list' or 'data_list' is not a list.
    ValueError: If 'time_list' and 'data_list' are not of equal length.
    """
    if not (isinstance(time_list, list) and isinstance(data_list, list)):
        raise TypeError("'time_list' and 'data_list' should be lists.")
    if len(time_list) != len(data_list):
        raise ValueError(
            "Length of 'time_list' and 'data_list' must be equal.")

    df = pd.DataFrame({'time': time_list, 'data': data_list})
    df.set_index('time', inplace=True)
    df_resampled = df.resample(interval).mean()

    return df_resampled.index.tolist(), df_resampled['data'].tolist()


def find_noon_and_midnight_time(time_diff, date_str, gk2a=False):
    """
    Find the local times for noon and midnight based on a given time
    difference and date.

    Parameters
    ----------
    time_diff (float): The time difference in hours.
    date_str (str): The date in 'YYYY-MM-DD' format.
    gk2a (bool): Flag to adjust time calculation for GK2A, defaults to False.

    Returns
    -------
    times (Tuple[dt.datetime, dt.datetime]): The calculated local times for
    noon and midnight.

    Raises
    ------
    TypeError: If 'time_diff' is not a number.
    ValueError: If 'date_str' is not in the correct format.
    """
    if not isinstance(time_diff, (int, float)):
        raise TypeError("'time_diff' must be a number.")
    try:
        date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(
            "Incorrect 'date_str' format, should be 'YYYY-MM-DD'.")

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

def mean_and_std_dev(data_1, data_2):
    """
    Calculate the mean difference and the standard deviation of the
    differences between two datasets while ignoring NaN values.

    Parameters
    ----------
    data_1 (List[float]): Data points of the first dataset.
    data_2 (List[float]): Data points of the second dataset.

    Returns
    -------
    stats (Tuple[float, float]): A tuple containing the mean difference and
    the standard deviation.

    Raises
    ------
    TypeError: If 'data_1' or 'data_2' is not a list.
    ValueError: If 'data_1' and 'data_2' lists are not of equal length.
    """
    # Convert lists to numpy arrays if they are not already
    if not (isinstance(data_1, list) and isinstance(data_2, list)):
        raise TypeError("Both 'data_1' and 'data_2' should be lists.")
    if len(data_1) != len(data_2):
        raise ValueError(
            "Length of 'data_1' and 'data_2' lists must be equal.")

    if isinstance(data_1, list):
        data_1 = np.array(data_1, dtype=np.float64)
    if isinstance(data_2, list):
        data_2 = np.array(data_2, dtype=np.float64)

    differences = data_1 - data_2

    # ignoring NaN values for mean and std:
    mean_diff = np.nanmean(differences)
    std_dev = np.nanstd(differences)

    return mean_diff, std_dev


def find_data_errors(data, window=5, threshold=5):
    """
    Finds erroneous data points by flagging outliers that are a certain number
    of median absolute deviations away from the median of their neighbors.

    Parameters:
    data (np.array): The dataset to be checked for errors.
    window (int): The number of neighboring points to consider for each check.
    threshold (float): The number of median absolute deviations from the median
                       which will be considered an outlier.

    Returns:
    List[int]: A list of indices where the data points are considered outliers.
    """

    if window % 2 == 0 or window < 1:
        raise ValueError("Window size must be odd and greater than 0")

    # Pad the data at the beginning and end to handle the window at edges
    padded_data = np.pad(data, (window // 2, window // 2), mode='median')

    # List to hold the indices of outliers
    outliers = []

    # Calculate the median and median absolute deviation
    for i in range(window // 2, len(data) + window // 2):
        # Create the windowed subset of the data
        window_data = padded_data[i - window // 2: i + window // 2 + 1]

        # Calculate the median of the window
        local_median = np.median(window_data)

        # Calculate the median absolute deviation
        mad = np.median(np.abs(window_data - local_median))

        # Check if the data point is an outlier based on the MAD
        if mad == 0:  # Avoid division by zero
            continue
        if np.abs(data[i - window // 2] - local_median) / mad > threshold:
            outliers.append(i - window // 2)

    return outliers


def format_units(units):
    """
    Format the units string for LaTeX-style exponentiation.

    Parameters:
    units (str): The original units string.

    Returns:
    str: The formatted units string.
    """
    formatted_units = re.sub(r'-(\d+)', r'^{-\1}', units)
    # Wrap each unit with LaTeX math mode symbols
    formatted_units = formatted_units.replace(' ', '$ $')
    return f'${formatted_units}$'

def fix_data_error_with_nan(data, index):
    """
    Marks an erroneous data point as NaN.

    Parameters:
    data (np.array): The dataset containing the erroneous point.
    index (int): The index of the erroneous data point to be marked as NaN.

    Returns:
    np.array: The dataset with the specified point set to NaN.
    """

    # Ensure the index is within the proper range
    if index < 0 or index >= len(data):
        raise ValueError("Index out of range.")

    # Set the erroneous point to NaN
    data[index] = np.nan
    return data

# Example usage:
# outliers = find_data_errors(goes16_data[:, 2])
# print("Outlier indices:", outliers)
