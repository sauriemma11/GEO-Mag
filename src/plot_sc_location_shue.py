import netCDF4 as nc
import argparse
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import spacepy.coordinates as spcoords
import spacepy.time as spt
import spacepy.omni as omni
from datetime import datetime, timedelta
import os

if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"
from plotter import plot_spacecraft_positions_with_earth_and_magnetopause

RE_EARTH = 6378
GEOSTAT = 6.6  # geostationary orbit - Re

from cdasws import CdasWs

cdas = CdasWs()
from cdasws.datarepresentation import DataRepresentation as dr

# I am using cdas to get omni data, so this is how I found what variables to
# grab
# datasets = cdas.get_datasets(observatoryGroup='OMNI', instrumentType='')
# print(datasets)

# instr = cdas.get_instruments(observatory='OMNI (1AU IP Data)')
# instr = cdas.get_instruments(observatoryGroup='OMNI (1AU IP Data)')
# print(instr)

# obs_groups = cdas.get_observatory_groups()
# for index, obs_group in enumerate(obs_groups):
#     print(obs_group['Name'])

# instr_types = cdas.get_instrument_types()
# for index, instr_type in enumerate(instr_types):
#     print(instr_type['Name'])

# datasets = cdas.get_datasets(observatoryGroup='OMNI (Combined 1AU IP Data)')
# for index, dataset in enumerate(datasets):
#     print(dataset['Id'], dataset['Label'])

# variables = cdas.get_variables('OMNI_HRO_1MIN')
# for variable in variables:
#     print(variable['Name'], " : ", variable['LongDescription'])
# # BZ_GSM, Pressure,
# print("...")
# variables = cdas.get_variables('OMNI2_H0_MRG1HR')
# for variable in variables:
#     print(variable['Name'], variable['LongDescription'])
# var_names = ['BZ_GSE1800', 'BZ_GSM1800', 'Pressure1800']

var_names = ['BZ_GSM', 'Pressure']
data = cdas.get_data('OMNI_HRO_1MIN', var_names, '2023-02-26T21:30:00Z',
                     '2023-02-26T21:59:00Z', dataRepresentation=dr.XARRAY)[1]


# print(data)
# print(type(data))
# print(data.BZ_GSM.Epoch.values) # Time
# print(data.BZ_GSM.values) # Data
# print(data.Pressure.values)


# observatories = cdas.get_observatories()

# for obs_info in observatories:
#     print(obs_info['Name'])

# OMNI_1min = 'OMNI_HRO_1MIN'
# variables_to_get = ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'BY_GSM', 'BZ_GSM',
# 'proton_density', 'Vx', 'Vy', 'Vz']
#
# data = cdas.get_data(OMNI_1min, variables=variables_to_get, time0=sdate_s,
# time1=edate_s)
# variables_in_dataset = data[1]


def parse_arguments():
    """
    Parse the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Pass in orbit .nc files")
    parser.add_argument('--g16',
                        type=str,
                        help="Path to g16 orb information (must be .nc file)",
                        required=False)

    parser.add_argument('--g17',
                        type=str,
                        help="Path to g17 orb information (must be .nc file)",
                        required=False)

    parser.add_argument('--g18',
                        type=str,
                        help="Path to g18 orb information (must be .nc file)",
                        required=False)

    # TODO: change file type for gk2a since orb location info is different
    parser.add_argument('--gk2a',
                        type=str,
                        help="Path to gk2a orb information",
                        required=False)

    parser.add_argument('--timestamp',
                        type=str,
                        help="time stamp (as a string) ex. YYYYMMDDHH",
                        required=True)

    return parser.parse_args()


def get_omni_data(date_str, minute):
    # convert timestamp to the spacepy ticktock format for OMNI data retrieval
    ticks = spt.Ticktock([timestamp], 'UTC')
    data = omni.get_omni(ticks)

    return data['BzIMF'][0], data['Pressure'][0]


def j2000_to_datetime(timestamp):  # for sosmag data
    epoch = pd.to_datetime('2000-01-01 00:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime


def goes_epoch_to_datetime(timestamp):  # for GOES data
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime


def gse_to_earth(pos, alpha=np.radians(23.5)):
    """
    Convert coordinates from GSE to Earth-centered.

    Parameters:
        pos (array-like): The position in GSE coordinates.
                          Should be an array-like structure with shape (n, 3).
        alpha (float): Rotation angle in radians. Default is 0.

    Returns:
        np.ndarray: The converted coordinates in Earth-centered system.
    """
    pos = np.array(pos)  # make sure input is a numpy array
    x_gse, y_gse, z_gse = pos[:, 0], pos[:, 1], pos[:, 2]

    x = x_gse * np.cos(alpha) + y_gse * np.sin(alpha)
    y = -x_gse * np.sin(alpha) + y_gse * np.cos(alpha)
    z = z_gse

    return np.column_stack((x, y, z))


def apply_GSE_nparraystack(pos):
    """
    Convert coordinates from GSE to Earth-centered.

    Parameters:
        pos (array-like): The position in GSE coordinates.
                          Should be an array-like structure with shape (n, 3).
        alpha (float): Rotation angle in radians. Default is 0.

    Returns:
        np.ndarray: The converted coordinates in Earth-centered system.
    """
    pos = np.array(pos)  # make sure input is a numpy array
    x_gse, y_gse, z_gse = pos[:, 0], pos[:, 1], pos[:, 2]

    x = x_gse
    y = y_gse
    z = z_gse

    return np.column_stack((x, y, z))


def apply_gse_to_earth_to_dict(coordinates_dict):
    transformed_coordinates = {}
    for satellite, coords in coordinates_dict.items():
        coord_values = np.array([[coords['X'], coords['Y'], coords['Z']]])
        earth_coords = apply_GSE_nparraystack(coord_values)

        # print(f"Original: {coords}, Transformed: {earth_coords}")  #
        # Debugging line

        transformed_coordinates[satellite] = {
            'X': earth_coords[0, 0],
            'Y': earth_coords[0, 1],
            'Z': earth_coords[0, 2]
        }
    return transformed_coordinates


def cartesian_to_polar(pos):
    x, y, z = pos
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.degrees(np.arctan2(y, x))
    return r, theta


def convert_GSE(spc_coords_file, hour):
    """
    Convert GSE coordinates to GSM coordinates and filter based on the
    specified hour.

    Parameters:
        spc_coords_file (str): Path to the spc_coords file (must be .nc file).
        hour (int): Hour value to filter the coordinates.

    Returns:
        pandas.DataFrame: DataFrame containing columns [time, Xgsm, Ygsm,
        Zgsm] with units being positional data cartesian GSM in RE.
    """

    spc_coords = nc.Dataset(spc_coords_file)
    spcCoords_time = goes_epoch_to_datetime(
        spc_coords['time'][:]).to_pydatetime().tolist()
    tickz = spt.Ticktock(spcCoords_time, 'UTC')

    x, y, z = spc_coords['gse_xyz'][:, 0], spc_coords['gse_xyz'][:, 1], \
    spc_coords['gse_xyz'][:, 2]
    pos_gse = np.column_stack((x, y, z))
    pos_gse_coords = spcoords.Coords(pos_gse, 'GSE', 'car', ticks=tickz)
    pos_x_gse, pos_y_gse, pos_z_gse = pos_gse_coords.x[:], pos_gse_coords.y[
                                                           :], \
        pos_gse_coords.z[
                                                               :]

    spc_coords_df = pd.DataFrame(
        {'time': spcCoords_time, 'X': pos_x_gse, 'Y': pos_y_gse,
         'Z': pos_z_gse})

    # Filter coordinates based on hour
    spc_coords_df['hour'] = spc_coords_df['time'].apply(lambda x: x.hour)
    filtered_coords_df = spc_coords_df[(spc_coords_df['hour'] == hour)].drop(
        columns=['hour'])

    return filtered_coords_df


def convertGSEkmGSMRe(spc_coords_file, hour):
    """
    Convert GSE coordinates to GSM coordinates and filter based on the
    specified hour.

    Parameters:
        spc_coords_file (str): Path to the spc_coords file (must be .nc file).
        hour (int): Hour value to filter the coordinates.

    Returns:
        pandas.DataFrame: DataFrame containing columns [time, Xgsm, Ygsm,
        Zgsm] with units being positional data cartesian GSM in RE.
    """

    spc_coords = nc.Dataset(spc_coords_file)
    spcCoords_time = goes_epoch_to_datetime(
        spc_coords['time'][:]).to_pydatetime().tolist()
    tickz = spt.Ticktock(spcCoords_time, 'UTC')

    x, y, z = spc_coords['gse_xyz'][:, 0], spc_coords['gse_xyz'][:, 1], \
    spc_coords['gse_xyz'][:, 2]
    pos_gse = np.column_stack((x, y, z))
    pos_gse_coords = spcoords.Coords(pos_gse, 'GSE', 'car', ticks=tickz)

    pos_gsm_coords = pos_gse_coords.convert('GSM', 'car')
    pos_x_gsm, pos_y_gsm, pos_z_gsm = pos_gsm_coords.x[:], pos_gsm_coords.y[
                                                           :], \
        pos_gsm_coords.z[
                                                               :]

    spc_coords_df = pd.DataFrame(
        {'time': spcCoords_time, 'Xgsm': pos_x_gsm, 'Ygsm': pos_y_gsm,
         'Zgsm': pos_z_gsm})

    # Filter coordinates based on hour
    spc_coords_df['hour'] = spc_coords_df['time'].apply(lambda x: x.hour)
    filtered_coords_df = spc_coords_df[(spc_coords_df['hour'] == hour)].drop(
        columns=['hour'])

    return filtered_coords_df


# test_with_g18 = convertGSEkmGSMRe(
# f'C:/Users/sarah.auriemma/Desktop/Data_new/g18/orb/2022_08/dn_ephe-l2
# -orb1m_g18_d{date_str_for_filesearch}_v0-0-3.nc', hour)
# print(test_with_g18)

def process_sat_data_inputs(args):
    """
        Process the satellite data files based on the provided command-line
        arguments.

        Parameters:
            args (argparse.Namespace): Parsed command-line arguments.

        Returns:
            dict: Dictionary containing satellite data with satellite names
            as keys.
    """
    timestamp = datetime.strptime(args.timestamp, '%Y%m%d%H')
    date_str_for_filesearch = args.timestamp[:8]
    hour = timestamp.hour

    satellites_data = {}
    satellites = {
        'g16': args.g16,
        'g17': args.g17,
        'g18': args.g18,
        'gk2a': args.gk2a
    }

    for satellite_name, satellite_file in satellites.items():
        if satellite_file:
            satellite_data = convert_GSE(satellite_file, hour)
            satellites_data[satellite_name] = satellite_data
            print(f"Data for {satellite_name}:")
            print(satellite_data)  # Print the data for each satellite
        else:
            print(f"No file provided for {satellite_name}.")

    return satellites_data


def user_selection_criteria(satellites_data, date_str, hour):
    """
    Prompt the user to enter minute(s) as selection criteria.

    Parameters:
        satellites_data (dict): Dictionary containing satellite data.
        date_str (str): Date string in 'YYYYMMDD' format.
        hour (int): Hour of the data.
    """

    user_input = input(
        "Enter a minute (e.g., '30') or a range of minutes (e.g., '15-45'): ")

    if '-' in user_input:
        start_minute, end_minute = user_input.split('-')
        # process_time_range(satellites_data, date_str, hour, start_minute,
        #                    end_minute)
        return process_time_range(satellites_data, date_str, hour,
                                  start_minute, end_minute)

    else:
        return process_single_minute(satellites_data, date_str, hour,
                                     user_input)
        # process_single_minute(satellites_data, date_str, hour, user_input)


def process_time_range(satellites_data, date_str, hour, start_minute,
                       end_minute):
    """
    Process and average the data within a specified time range.

    Parameters:
        satellites_data (dict): Dictionary containing satellite data.
        date_str (str): Date string in 'YYYYMMDD' format.
        hour (int): Hour of the data.
        start_minute (str): Start minute of the range.
        end_minute (str): End minute of the range.
    """
    start_timestamp = pd.to_datetime(f"{date_str} {hour}:{start_minute}:00")
    end_timestamp = pd.to_datetime(f"{date_str} {hour}:{end_minute}:00")

    coordinates_dict = {}

    for satellite_name, data in satellites_data.items():
        data['time'] = pd.to_datetime(data['time'])

        # filter data based on the timestamp range
        filtered_data = data[(data['time'] >= start_timestamp) & (
                    data['time'] <= end_timestamp)]

        # make sure filtered data is not empty before computing mean
        if not filtered_data.empty:
            average_data = filtered_data.mean()
            coordinates_dict[satellite_name] = average_data[
                ['X', 'Y', 'Z']].to_dict()

    print(coordinates_dict)
    return coordinates_dict


def process_single_minute(satellites_data, date_str, hour, minute):
    """
    Output the data for each satellite at a single minute.

    Parameters:
        satellites_data (dict): Dictionary containing satellite data.
        date_str (str): Date string in 'YYYYMMDD' format.
        hour (int): Hour of the data.
        minute (str): The specified minute.
    """
    target_timestamp = pd.to_datetime(f"{date_str} {hour}:{minute}:00")

    coordinates_dict = {}

    for satellite_name, data in satellites_data.items():

        data['time'] = pd.to_datetime(data['time'])

        specific_data = data[data['time'] == target_timestamp]
        if not specific_data.empty:
            specific_data = specific_data.iloc[0]
            coordinates_dict[satellite_name] = specific_data[
                ['X', 'Y', 'Z']].to_dict()

    print(coordinates_dict)
    return coordinates_dict


def main():
    args = parse_arguments()
    satellites_data = process_sat_data_inputs(args)

    date_str = args.timestamp[:8]
    hour = datetime.strptime(args.timestamp, '%Y%m%d%H').hour
    timestamp_for_OMNI = datetime.strptime(args.timestamp, '%Y%m%d%H')

    # User selects data based on their criteria
    coordinates_dict = user_selection_criteria(satellites_data, date_str, hour)

    # Transform the selected coordinates from GSE to Earth-centered system
    # transformed_dict = apply_gse_to_earth_to_dict(coordinates_dict)
    transformed_dict = apply_gse_to_earth_to_dict(coordinates_dict)

    # for satellite, coords in transformed_dict.items():
    #     print(f"{satellite}: {coords}")

    # imf_bz, solar_wind_pressure = get_omni_data(timestamp_for_OMNI)
    imf_bz, solar_wind_pressure = -12, 6.0

    # Now, you can use the plotting function from plotting.py
    plot_spacecraft_positions_with_earth_and_magnetopause(transformed_dict,
                                                          solar_wind_pressure,
                                                          imf_bz)


if __name__ == "__main__":
    main()
