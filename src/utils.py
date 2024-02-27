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


# Below are timestamp utils from Brian Kress:

def timestamp_constants():
    """
    Provides certain constants related to time stamps.

    Returns
    -------
    J2000_EPOCH : datetime.datetime
        J2000 epoch as a datetime object.
    SECONDS_POSIX_J2000 : float
        Number of seconds from POSIX time to J2000 epoch.
    N_SECONDS_DAY : int
        Number of seconds in a POSIX day.
    """
    N_SECONDS_DAY = 86400  # Correct for a POSIX day, even one with a leap
    # second.
    J2000_EPOCH = dt.datetime(2000, 1, 1, 12, 0, 0)
    SECONDS_POSIX_J2000 = 946728000.0

    return J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY


def j2000_to_posix(j2000_seconds):
    """
    Converts vector of J2000 time stamps (in seconds) to Datetime POSIX
    timestamps
    """

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = timestamp_constants()

    dt_list = []

    posix_seconds = SECONDS_POSIX_J2000 + j2000_seconds
    npts = len(posix_seconds)

    for i in range(npts):
        # dt64 = np.datetime64(datetime.datetime.utcfromtimestamp(
        #     posix_seconds[i]))
        # dt_list.append(dt64)
        dtt = dt.datetime.utcfromtimestamp(posix_seconds[i])
        dt_list.append(dtt)

    return np.array(dt_list)


def j2000_to_posix_0d(j2000_seconds):
    """DESCRIPTION:
      Converts J2000 time stamp scalar (in seconds) to Datetime POSIX timestamp
    MODULES:
      datetime
    INPUTS:
      j2000_seconds: seconds since J2000 epoch
    OUTPUTS:
      datetime"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = timestamp_constants()

    posix_seconds = SECONDS_POSIX_J2000 + j2000_seconds

    dtt = dt.datetime.utcfromtimestamp(posix_seconds)

    return dtt


def j2000_1s_timestamps(year, month, date, N_REPORTS):
    """DESCRIPTION:
      Creates 1-second-cadence timestamps for the given date in a 2-d array
      with one dimension being the number of seconds in N_REPORTS.  Timestamps
      represent the number of seconds since the J2000 epoch.
    MODULES:
      datetime, numpy
    INPUTS:
      year, month, date: integers describing date (UT)
      N_REPORTS: number of seconds in L1b record -- likely 30 or 60
    OUTPUTS:
      timestamp_array:  n_files x N_REPORTS NumPy array containing timestamps
      measured in number of seconds since 01 Jan 2000, 1200 UT"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    n_files = N_SECONDS_DAY / N_REPORTS

    timestamp_array = np.zeros((n_files, N_REPORTS))

    dt_start = dt.datetime(year, month, date, 0, 0, 0)

    time_stamp = (dt_start - J2000_EPOCH).total_seconds()

    for r in range(n_files):
        for s in range(N_REPORTS):
            timestamp_array[r, s] = time_stamp
            time_stamp += 1.0

    return timestamp_array


def j2000_p1s_timestamps(year, month, date, N_REPORTS, SAMPLES_PER_REPORT):
    """DESCRIPTION:
      Creates 0.1-second-cadence timestamps for the given date in a 3-d array
      with one dimension being the number of seconds in N_REPORTS.  Timestamps
      represent the number of seconds since the J2000 epoch.
    MODULES:
      datetime, numpy
    INPUTS:
      year, month, date: integers describing date (UT)
      N_REPORTS: number of seconds in L1b record -- likely 60
      SAMPLES_PER_REPORT: number of samples in one second: likely 10
    OUTPUTS:
      timestamp_array:  n_files x N_REPORTS x SAMPLES_PER_REPORT NumPy array
      containing timestamps measured in number of seconds since 01 Jan 2000,
      1200 UT"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    n_files = N_SECONDS_DAY / N_REPORTS
    dtt = np.float64(1) / np.float64(SAMPLES_PER_REPORT)

    timestamp_array = np.zeros((n_files, N_REPORTS, SAMPLES_PER_REPORT),
                               dtype='float64')

    dt_start = dt.datetime(year, month, date, 0, 0, 0)

    time_stamp = np.float64((dt_start - J2000_EPOCH).total_seconds())

    for r in range(n_files):
        for s in range(N_REPORTS):
            for t in range(SAMPLES_PER_REPORT):
                timestamp_array[r, s, t] = time_stamp
                time_stamp += dtt

    return timestamp_array


def create_j2000_timestamps(year, month, date, n_seconds_cadence):
    """
    Creates n_seconds_cadence timestamps for a given date.

    Parameters
    ----------
    year : int
        The year.
    month : int
        The month.
    date : int
        The day of the month.
    n_seconds_cadence : int
        Cadence in number of seconds.

    Returns
    -------
    timestamp_array : numpy.ndarray
        Array containing timestamps measured in seconds since 01 Jan 2000,
        1200 UT.
    """
    J2000_EPOCH, _, N_SECONDS_DAY = timestamp_constants()

    n_timestamps = int(N_SECONDS_DAY / n_seconds_cadence)

    timestamp_array = np.zeros(n_timestamps)

    dt_start = dt.datetime(year, month, date, 0, 0, 0)
    time_stamp = (dt_start - J2000_EPOCH).total_seconds()

    for r in range(n_timestamps):
        timestamp_array[r] = time_stamp
        time_stamp += n_seconds_cadence

    return timestamp_array


def posix_to_j2000(year, month, day, hour, min, sec):
    """DESCRIPTION:
      Converts Datetime POSIX timestamp to J2000 time stamp (in seconds)
    MODULES:
      datetime
    INPUTS:
      NumPy array containing datetimes
    OUTPUTS:
      j2000_seconds: NumPy array of seconds since J2000 epoch"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    dt_start = dt.datetime(year, month, day, hour, min, sec)

    timestamp = (dt_start - J2000_EPOCH).total_seconds()

    return timestamp


def iso8601_to_datetime(iso8601):
    """DESCRIPTION:
      Converts an ISO8601 string (e.g. '2010-04-05T09:30:00.0Z') to a datetime
      object
    MODULES:
      datetime
    INPUTS:
      iso8601: string
    OUTPUTS:
      dt: datetime object
      j2000_sec: seconds since J2000 epoch"""

    year = int(iso8601[0:4])
    month = int(iso8601[5:7])
    day = int(iso8601[8:10])
    hour = int(iso8601[11:13])
    min = int(iso8601[14:16])
    sec = int(iso8601[17:19])
    micro = int(iso8601[20:21]) * 100000

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    dtt = dt.datetime(year, month, day, hour, min, sec, micro)

    j2000_sec = (dtt - J2000_EPOCH).total_seconds()

    return dtt, j2000_sec


def doy_to_dom(year, doy):
    """DESCRIPTION:
      This function converts day of year (doy) and year to number of month and
      day of month (dom)
    MODULES:
      NumPy
    INPUTS:
      year: 4-digit year, e.g. 2000
      doy: 1-366
    OUTPUTS:
      month: 1-12
      dom: 1-31"""

    if (year % 4) != 0:
        days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    else:
        days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    days = np.array(days)
    month = days[days < doy].shape[0]
    dom = doy - days[month - 1]

    return month, dom


def dom_to_doy(year, month, dom):
    """DESCRIPTION:
      This function converts day of year (doy) and year to number of month and
      day of month (dom)
    MODULES:
      NumPy
    INPUTS:
      year: 4-digit year, e.g. 2000
      doy: 1-366
    OUTPUTS:
      month: 1-12
      dom: 1-31"""

    if (year % 4) != 0:
        days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    else:
        days = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

    doy = days[month - 1] + dom

    return doy


def doy_to_j2000(year, doy):
    """DESCRIPTION:
      This function converts day of year (doy) and year to seconds since J2000
      epoch at 00 UT
    MODULES:
      NumPy, datetime
    INPUTS:
      year: 4-digit year, e.g. 2000
      doy: 1-366
    OUTPUTS:+
      j2000_start: seconds since J2000 epoch at start of given doy"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    (month, dom) = doy_to_dom(year, doy)
    dt_start = dt.datetime(year, month, dom, 0, 0, 0)

    j2000_start = (dt_start - J2000_EPOCH).total_seconds()

    return j2000_start


def doy_to_j2000_day(year, doy):
    """DESCRIPTION:
      Determines number of integral days since J2000 epoch and
      milliseconds at start of current UT day.  To simulate L0 header times.
    MODULES:
      NumPy, datetime
    INPUTS:
      year: 4-digit year, e.g. 2000
      doy: 1-366
    OUTPUTS:+
      j2000_start: seconds since J2000 epoch at start of given doy"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()
    j2000_start = doy_to_j2000(year, doy)

    n_day_j2k = np.int64(j2000_start / N_SECONDS_DAY)
    ms_j2k = (j2000_start - n_day_j2k * N_SECONDS_DAY) * 1000.0

    return n_day_j2k, ms_j2k


def j2000_to_doy(j2000_seconds):
    """DESCRIPTION:
      This function converts seconds since J2000 epoch at 00 UT
      to day of year
    MODULES:
      NumPy, datetime
    INPUTS:
      year: 4-digit year, e.g. 2000
      doy: 1-366
    OUTPUTS:+
      j2000_start: seconds since J2000 epoch at start of given doy"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    posix_seconds = SECONDS_POSIX_J2000 + j2000_seconds

    dtt = dt.datetime.utcfromtimestamp(posix_seconds)

    doy = dom_to_doy(dtt.year, dtt.month, dtt.day)

    return np.int32(doy)


def j2000_to_iso8601(j2000_seconds):
    """DESCRIPTION:
      This function converts a floating point scalar quantifying the number of
      seconds since J2000 epoch to an ISO 8601 string (UT).
    MODULES:
      datetime
    INPUTS:
      j2000_seconds: seconds since J2000 epoch, float
    OUTPUTS:
      iso8601: ISO8601 string representing UT"""

    (J2000_EPOCH, SECONDS_POSIX_J2000, N_SECONDS_DAY) = \
        timestamp_constants()

    dtt = dt.datetime.utcfromtimestamp(SECONDS_POSIX_J2000 +
                                       j2000_seconds)

    iso8601 = dtt.isoformat()

    return iso8601
