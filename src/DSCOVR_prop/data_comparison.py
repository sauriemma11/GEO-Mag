import pandas as pd
import numpy as np
import datetime as dt
import json
from dscovr_propagation import propagate_parameters
import glob
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from icecream import ic


def store_data_hdf5(df, file_path):
    df.to_hdf(file_path, 'omni_data', format='table', data_columns=True, complib='blosc', complevel=9)


def read_data_hdf5(file_path):
    return pd.read_hdf(file_path, 'omni_data')


def parse_omni_data(file_path):
    """Parse fixed-width OMNI data file into DataFrame with cleaned data and converted datetime."""
    col_names = [
        'Year', 'Day', 'Hour', 'Minute',
        'BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM',
        'Flow_Speed', 'Vx_Velocity', 'Proton_Density'
    ]
    col_widths = [4, 4, 3, 3, 8, 8, 8, 8, 8, 8, 8, 7]

    df = pd.read_fwf(file_path, widths=col_widths, names=col_names, skiprows=1)
    placeholder_values = {9999.99: np.nan, 99999.9: np.nan, 999.99: np.nan}
    df.replace(placeholder_values, inplace=True)

    df[['Year', 'Day', 'Hour', 'Minute']] = df[['Year', 'Day', 'Hour', 'Minute']].apply(pd.to_numeric, errors='coerce',
                                                                                        downcast='integer')
    df.dropna(subset=['Year', 'Day', 'Hour', 'Minute'], inplace=True)

    df['DateTime'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Day'].astype(int).astype(str),
                                    format='%Y-%j') + pd.to_timedelta(df['Hour'].astype(int),
                                                                      unit='h') + pd.to_timedelta(
        df['Minute'].astype(int), unit='m')

    df.drop(['Year', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)

    df.rename(columns={
        'BX_GSE': 'bx_gse',
        'BY_GSE': 'by_gse',
        'BZ_GSE': 'bz_gse',
        'BY_GSM': 'by_gsm',
        'BZ_GSM': 'bz_gsm',
        'Flow_Speed': 'proton_speed',
        'Vx_Velocity': 'proton_vx_gsm',
        'Proton_Density': 'proton_density'
    }, inplace=True)

    return df


def verify_file_accessibility(patterns):
    for pattern in patterns:
        files = glob.glob(pattern)
        if not files:
            print(f"No files found for pattern: {pattern}")
        for file_path in files:
            try:
                with Dataset(file_path, mode='r') as file:
                    print(f"Successfully accessed {file_path}")
            except Exception as e:
                print(f"Failed to access {file_path}. Error: {e}")


def generate_daily_file_paths(date):
    base_path = "Z:/Data/DSCOVR"
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')

    date_folder = os.path.join(base_path, year)

    start_time = f"{year}{month}{day}000000"

    mag_file_pattern = f"oe_m1m_dscovr_s{start_time}_*.nc"
    sw_file_pattern = f"oe_f1m_dscovr_s{start_time}_*.nc"
    pos_file_pattern = f"oe_pop_dscovr_s{start_time}_*.nc"

    mag_path = os.path.join(date_folder, mag_file_pattern)
    sw_path = os.path.join(date_folder, sw_file_pattern)
    pos_path = os.path.join(date_folder, pos_file_pattern)

    return mag_path, sw_path, pos_path


def generate_daily_config(date, base_config):
    config = dict(base_config)
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')

    base_path = "Z:/Data/DSCOVR"
    start_time = f"{year}{month}{day}000000"

    mag_file_pattern = os.path.join(base_path, year, f"oe_m1m_dscovr_s{start_time}_*.nc")
    sw_file_pattern = os.path.join(base_path, year, f"oe_f1m_dscovr_s{start_time}_*.nc")
    pos_file_pattern = os.path.join(base_path, year, f"oe_pop_dscovr_s{start_time}_*.nc")

    mag_files = glob.glob(mag_file_pattern)
    sw_files = glob.glob(sw_file_pattern)
    pos_files = glob.glob(pos_file_pattern)

    if not mag_files or not sw_files or not pos_files:
        raise FileNotFoundError(f"Files not found for date {date.strftime('%Y-%m-%d')}")

    config['files']['mag_file'] = mag_files[0]
    config['files']['sw_file'] = sw_files[0]
    config['files']['pos_file'] = pos_files[0]

    config_path = f"temp_config_{date.strftime('%Y%m%d')}.json"
    with open(config_path, 'w') as file:
        json.dump(config, file)

    return config_path


def propagate_data_over_period(start_date, end_date, base_config_path):
    with open(base_config_path, 'r') as file:
        base_config = json.load(file)

    current_date = start_date
    all_results = []

    while current_date <= end_date:
        print(f"Propagating data for {current_date.strftime('%Y-%m-%d')}")
        daily_config_path = generate_daily_config(current_date, base_config)

        daily_result = propagate_parameters(daily_config_path)

        if 'time' not in daily_result:
            daily_result['time'] = [current_date + dt.timedelta(minutes=i) for i in range(len(daily_result['bx_gse']))]

        all_results.append(daily_result)

        os.remove(daily_config_path)
        current_date += dt.timedelta(days=1)

    return all_results


def round_and_aggregate(data, round_to='min'):
    df = pd.DataFrame(data)

    if 'time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['time'])
    elif 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    else:
        raise KeyError("The dataframe does not contain 'time' or 'DateTime' column.")

    df.set_index('DateTime', inplace=True)
    df = df.resample(round_to).mean().reset_index()
    return df


def compare_omni_and_propagated(omni_data, propagated_data):
    if 'DateTime' not in omni_data.columns:
        omni_data = omni_data.reset_index()

    if 'DateTime' not in propagated_data.columns:
        propagated_data = propagated_data.reset_index()

    # TODO: trim omni data by day first before comparing

    start_time = propagated_data['DateTime'].min()
    end_time = propagated_data['DateTime'].max()

    omni_data_trimmed = omni_data[(omni_data['DateTime'] >= start_time) & (omni_data['DateTime'] <= end_time)]

    # Merge on DateTime
    merged_df = pd.merge(omni_data_trimmed, propagated_data, on='DateTime', suffixes=('_omni', '_prop'))

    # Calculate differences
    differences = {}
    for column in ['bx_gse', 'by_gse', 'bz_gse', 'by_gsm', 'bz_gsm', 'proton_speed', 'proton_density', 'proton_vx_gsm']:
        differences[column] = merged_df[f'{column}_omni'] - merged_df[f'{column}_prop']

    return differences


def calculate_statistics(differences):
    stats = {}
    for key, value in differences.items():
        stats[key] = {
            'mean': np.nanmean(value),
            'median': np.nanmedian(value),
            'std_dev': np.nanstd(value)
        }
    return stats


def plot_differences(differences):
    keys = list(differences.keys())
    n = len(keys)
    fig, axs = plt.subplots(2, 4)  # Create a 2x4 grid
    fig.tight_layout(pad=4.0)

    # Define units for each variable
    units = {
        'bx_gse': 'nT',
        'by_gse': 'nT',
        'bz_gse': 'nT',
        'by_gsm': 'nT',
        'bz_gsm': 'nT',
        'proton_speed': 'km/s',
        'proton_density': 'cm^-3',
        'proton_vx_gsm': 'km/s'
    }

    for i, key in enumerate(keys):
        row = i // 4
        col = i % 4
        axs[row, col].hist(differences[key], bins=50, alpha=0.7)
        axs[row, col].set_title(f'Histogram of Differences for {key} ({units.get(key, "")})')
        axs[row, col].set_xlabel(f'Difference ({units.get(key, "")})')
        axs[row, col].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(n, 8):
        row = j // 4
        col = j % 4
        fig.delaxes(axs[row, col])

    plt.show()


def calculate_covariance(omni_data, propagated_data):
    """Calculate covariance between OMNI and propagated data."""
    omni_data.set_index('DateTime', inplace=True)
    propagated_data.set_index('DateTime', inplace=True)

    covariance_matrices = {}
    for column in ['bx_gse', 'by_gse', 'bz_gse', 'by_gsm', 'bz_gsm', 'proton_speed', 'proton_density', 'proton_vx_gsm']:
        merged = pd.concat([omni_data[column], propagated_data[column]], axis=1).dropna()
        covariance_matrices[column] = np.cov(merged.values.T)

    return covariance_matrices


def plot_covariance(covariance_matrices):
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # Create a 2x4 grid
    fig.tight_layout(pad=4.0)

    keys = list(covariance_matrices.keys())
    for i, key in enumerate(keys):
        row = i // 4
        col = i % 4
        cax = axs[row, col].imshow(covariance_matrices[key], cmap='coolwarm', vmin=-1, vmax=1)
        axs[row, col].set_title(f'Covariance Matrix for {key}')
        fig.colorbar(cax, ax=axs[row, col], orientation='vertical')

    plt.show()


def plot_time_series(omni_data, propagated_data):
    keys = ['bx_gse', 'by_gse', 'bz_gse', 'by_gsm', 'bz_gsm', 'proton_speed', 'proton_density', 'proton_vx_gsm']
    units = {
        'bx_gse': 'nT',
        'by_gse': 'nT',
        'bz_gse': 'nT',
        'by_gsm': 'nT',
        'bz_gsm': 'nT',
        'proton_speed': 'km/s',
        'proton_density': 'cm^-3',
        'proton_vx_gsm': 'km/s'
    }

    if 'DateTime' not in omni_data.columns:
        omni_data['DateTime'] = pd.to_datetime(omni_data.index)

    if 'DateTime' not in propagated_data.columns:
        propagated_data['DateTime'] = pd.to_datetime(propagated_data.index)

    for key in keys:
        plt.figure(figsize=(10, 5))
        plt.plot(omni_data['DateTime'], omni_data[key], label=f'OMNI {key}', alpha=0.7)
        plt.plot(propagated_data['DateTime'], propagated_data[key], label=f'Propagated {key}', alpha=0.7)
        plt.title(f'Time Series for {key} ({units.get(key, "")})')
        plt.xlabel('DateTime')
        plt.ylabel(f'{key} ({units.get(key, "")})')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # file_path = '../../data/ascii.lst.txt'
    hdf5_path = '../../data/processed_data.h5'

    # omni_data = parse_omni_data(file_path)
    # store_data_hdf5(omni_data, hdf5_path)

    omni_data = read_data_hdf5(hdf5_path)

    # if 'DateTime' not in omni_data.columns:
    #     omni_data['DateTime'] = pd.to_datetime(omni_data.index)

    start_date = dt.datetime(2018, 2, 20)
    end_date = dt.datetime(2018, 2, 23)
    base_config_path = 'comparison_config.JSON'  # Path to your base configuration JSON file

    all_differences = {key: [] for key in
                       ['bx_gse', 'by_gse', 'bz_gse', 'by_gsm', 'bz_gsm', 'proton_speed', 'proton_density',
                        'proton_vx_gsm']}

    # Propagate data over the defined period
    prop_results = propagate_data_over_period(start_date, end_date, base_config_path)

    for day_result in prop_results:
        rounded_data = round_and_aggregate(day_result)

        differences = compare_omni_and_propagated(omni_data, rounded_data)

        for key in all_differences.keys():
            all_differences[key].extend(differences[key])

    statz = calculate_statistics(all_differences)
    cov_matrix = calculate_covariance(omni_data, rounded_data)
    # plot_covariance(cov_matrix)
    # plot_differences(all_differences)
    ic(len(omni_data), len(rounded_data))

    # plot_time_series(omni_data, rounded_data)

    ic(omni_data, rounded_data)
