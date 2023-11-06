import pandas as pd
import numpy as np
import pickle


def readKpData(csv_file):
    """
    Read .txt file for kp data, delimited by a tab.
    Columns are YYYY, M, D, H, Kp
    Example input data: '2019	4	1	9	2.667'
    :param csv_file: file location
    :return: dataframe
    """
    column_names = ['YYYY', 'M', 'D', 'H', 'Kp']
    df = pd.read_csv(csv_file, names=column_names, sep='\t')

    # Combine time columns into a datetime column
    df['Time'] = pd.to_datetime(
        df[['YYYY', 'M', 'D', 'H']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H')

    # Drop the individual date and time columns
    df = df.drop(columns=['YYYY', 'M', 'D', 'H'])
    return df


def filterKpData(df, max_kp):
    """
    :param df: dataframe with Kp data
    :param max_kp: maximum Kp value for filtering
    :return: filtered dataframe, containing only values below max_kp
    """
    filtered_df = df[(df['Kp'] <= max_kp)]
    return filtered_df


def averageKpofDataset(df):
    """
    Takes dataframe as input, returns average kp over the timespan
    :param df:
    :return:
    """
    mean_val = df.loc[:, 'Kp'].mean()
    return mean_val


def replace_kp_above_max_with_nan(df, max_kp):
    """
    Replace Kp values above max_kp with NaN and create a boolean mask.

    :param filtered_data: DataFrame with Kp data
    :param max_kp: Maximum Kp value for filtering
    :return: Tuple containing (modified DataFrame, boolean mask)
    """
    # Create a mask for values above max_kp
    mask = df['Kp'] > max_kp

    # Replace values above max_kp with NaN
    df['Kp'][mask] = np.nan

    return df, mask


def set_subtr_to_nan_where_kp_over(subtr_list, kp_mask):
    """
    Set subtraction data to NaN where the mask is True.

        Args:
        subtr_list (list): List of subtraction data.
        kp_mask (list): Mask where True indicates time periods for NaN values.

    Returns:
        list: Updated subtraction data with NaN values where the mask is True.
    """
    # Update the subtraction data with NaN values where the mask is True
    updated_subtr_list = [subtr if not kp_mask[i] else np.nan for i, subtr in
                          enumerate(subtr_list)]
    return updated_subtr_list


def createkpMask(dfkp, datetime_list, max_kp=2):
    """
    Create a mask to identify time periods where Kp values are greater than
    a specified threshold.

    Args:
        dfkp (pd.DataFrame): DataFrame with Kp data.
        datetime_list (list): List of datetime values for subtraction data.
        max_kp (float): Maximum Kp value for filtering (default: 2).

    Returns:
        list: Mask where True indicates time periods where Kp is over max_kp.
    """
    # Initialize the mask as False for the length of datetime_list
    kp_mask = [False] * len(datetime_list)

    # Find Kp values greater than max_kp
    kp_over_threshold = dfkp['Kp'] > max_kp

    # Iterate over Kp data and extend the mask by three hours for each Kp
    # value over max_kp
    for i, kp_over in enumerate(kp_over_threshold):
        if kp_over:
            # Get the corresponding timestamp in datetime_list if it exists
            kp_timestamp = dfkp['Time'].iloc[i]
            try:
                # Find the index of this timestamp in datetime_list
                timestamp_index = datetime_list.index(kp_timestamp)

                # Set the mask to True for the next three hours (3-hour
                # cadence)
                kp_mask[timestamp_index:timestamp_index + 180] = [
                                                                     True] *\
                                                                 180  # 180
                # minutes = 3 hours
            except ValueError:
                # Handle the case where the timestamp is not found in
                # datetime_list
                pass

    return kp_mask


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


def calc_daily_stddev(datetime_list, subtr_list, kp_mask=None):
    """
    Calculate daily standard deviation of subtraction data with an optional
    Kp mask.

    Args:
        datetime_list (list): List of datetime values.
        subtr_list (list): List of subtraction data.
        kp_mask (list, optional): List of boolean values indicating Kp
        values over threshold.
                                  If provided, only data points where
                                  kp_mask is True will be used.

    Returns:
        pd.Series: daily standard deviation of subtraction data.
    """
    # Create a pandas DataFrame with datetime as the index
    df = pd.DataFrame({'datetime': datetime_list, 'subtraction': subtr_list})
    df.set_index('datetime', inplace=True)

    # Apply the Kp mask if provided
    if kp_mask is not None:
        df['subtraction'] = df['subtraction'].mask(kp_mask)

    # Resample the data to hourly frequency and calculate standard deviation
    daily_std_dev = df['subtraction'].resample('D').std()

    return daily_std_dev


def mean_std_dev(stddev):
    meanstddev = stddev.mean()
    return meanstddev


def mean_std_dev(stddev):
    meanstddev = stddev.mean()
    return meanstddev
