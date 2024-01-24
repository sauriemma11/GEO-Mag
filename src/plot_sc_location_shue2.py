import netCDF4 as nc
import argparse
import numpy as np
import pandas as pd
import spacepy.coordinates as spcoords
import spacepy.time as spt
from datetime import datetime, timedelta
from icecream import ic
import os
import pytplot
from pyspedas import sosmag_load

if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"
from plotter import plot_spacecraft_positions_with_earth_and_magnetopause

# GEOSTAT = 42164  # Radius of geostationary orbit in km (from Earth's center)

RE_EARTH = 6371  # Radius of Earth in km
GEOSTAT = 6.6  # geostationary orbit - Re
GK2A_LONG = 128.2  # GK2A consistent longitude, EAST
G18_LONG = 137.0  # WEST

from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation as dr

cdas = CdasWs()


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

    # Providing the orb info manually:
    # parser.add_argument('--gk2a',
    #                     type=str,
    #                     help="Path to gk2a orb information",
    #                     required=False)

    # Script will use pyspedas to get orb info:
    parser.add_argument("--gk2a", action='store_true',
                        help="Flag to indicate whether to plot gk2a")

    parser.add_argument('--timestamp',
                        type=str,
                        help="time stamp (as a string) ex. YYYYMMDD HH:MM",
                        required=True)

    return parser.parse_args()


def get_omni_values(date_str, hour, startminute, endminute):
    """
    Get BZ_imf and solar wind pressure from OMNI data.

    Parameters:
        date_str (str): Date in 'YYYY-MM-DD' format.
        time_input (int or tuple): Single minute or a range of minutes.
        is_single_minute (bool): Flag indicating if the input is a single
        minute.

    Returns:
        tuple: (BZ_imf, solar_wind_pressure)
    """

    if startminute == endminute:
        start_time = f"{date_str}T{hour}:{startminute}:00Z"
        end_time = f"{date_str}T{hour}:{endminute + 1}:00Z"
    else:
        start_time = f"{date_str}T{hour}:{startminute}:00Z"
        end_time = f"{date_str}T{hour}:{endminute}:00Z"

    data = cdas.get_data('OMNI_HRO_1MIN', ['BZ_GSM', 'Pressure'], start_time,
                         end_time, dataRepresentation=dr.XARRAY)[1]

    pressure_values = data.Pressure.values
    average_pressure = np.nanmean(pressure_values)
    max_pressure = np.nanmax(pressure_values)
    min_pressure = np.nanmin(pressure_values)

    print('BZ_imf: ')
    print(data.BZ_GSM.values)
    print('Pressure: ')
    print(pressure_values)

    print(f"Average Pressure: {average_pressure}")
    print(f"Maximum Pressure: {max_pressure}")
    print(f"Minimum Pressure: {min_pressure}")

    bz_imf = data.BZ_GSM.values[
        0] if 'BZ_GSM' in data and data.BZ_GSM.values.size > 0 else np.nan
    sw_pressure = data.Pressure.values[
        0] if 'Pressure' in data and data.Pressure.values.size > 0 else np.nan

    return bz_imf, sw_pressure


def load_sosmag_positional_data(timestamp_str):
    """
    Load and return SOSMAG/GK2A positional data for a specific timestamp.

    Parameters:
    timestamp_str (str): Timestamp in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
    pandas.DataFrame: Dataframe containing the positional data. Has columns:
    'X', 'Y', 'Z'. Index time.
    """
    # Parse the timestamp up to minutes
    timestamp = datetime.strptime(timestamp_str, '%Y%m%d %H:%M')

    # Manually append :00 for seconds
    start_time_str = timestamp.strftime('%Y%m%d %H:%M') + ':00'
    end_time_str = (timestamp + timedelta(minutes=1)).strftime(
        '%Y%m%d %H:%M') + ':00'

    # Convert back to datetime objects for the time range
    start_time = datetime.strptime(start_time_str, '%Y%m%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y%m%d %H:%M:%S')

    trange = [start_time.strftime('%Y-%m-%d %H:%M:%S'),
              end_time.strftime('%Y-%m-%d %H:%M:%S')]

    sosmag_load(trange=trange, datatype='1m')

    data_types = list(pytplot.data_quants.keys())
    data_types_array = np.array(data_types)

    # 'pos' is the positional data and always at the same index
    positional_str_pytplot = data_types_array[2] if len(
        data_types_array) > 2 else None

    if positional_str_pytplot:
        # Extracting data from pytplot's data structure
        positional_data = pytplot.data_quants[
            positional_str_pytplot].to_pandas()
        positional_data.rename(columns={0: 'X', 1: 'Y', 2: 'Z'}, inplace=True)
        # positional_data is a dataframe with columns 'X' 'Y' 'Z'

        print(f'Loaded GK2A positional data for {timestamp_str}')
        # return positional_data[['X', 'Y', 'Z']]  # Selecting only 'X',
        # 'Y', and 'Z' columns
        # print('debug', positional_data, type(positional_data))
        return positional_data
    else:
        print(f"GK2A positional data not available for {timestamp_str}")
        return None


def average_of_minute_timestamp_of_GK2A(df):
    """
    Calculate the average of the first and the middle data points in a
    DataFrame.

    This function is designed to work with a DataFrame where each row
    represents
    a timestamp of GK2A positional data. It calculates the average of the data
    at the first timestamp and the data at the middle timestamp of the
    DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the positional data,
                           with each row representing a timestamp.

    Returns:
    np.ndarray: A NumPy array containing the average of the first and middle
    data points.
                Returns None if the DataFrame is empty.

    Note:
    The function assumes that the DataFrame has an even number of rows. If
    the number
    of rows is odd, the function will use the lower middle row for the
    calculation.
    """
    if df.empty:
        print("The DataFrame is empty.")
        return None

    length_of_df = len(df.index)
    halfway_of_df = length_of_df // 2

    first_timestamp_data = df.iloc[0].to_numpy()
    middle_timestamp_data = df.iloc[halfway_of_df].to_numpy()
    average_data = (first_timestamp_data + middle_timestamp_data) / 2

    return average_data


def goes_epoch_to_datetime(timestamp):  # for GOES data
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime


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


def convert_and_filter_gse_by_timestamp(spc_coords_file, timestamp):
    """
    Convert GSE (Geocentric Solar Ecliptic) coordinates to GSM (Geocentric
    Solar Magnetospheric)
    coordinates and filter the data based on a specific timestamp.

    This function reads GSE coordinates from a .nc (NetCDF) file, converts
    them to GSM coordinates,
    and filters these coordinates to include data only at the specified
    timestamp.

    Parameters:
        spc_coords_file (str): Path to the spc_coords file (must be a .nc
        file).
        timestamp_str (str): Timestamp in the format 'YYYYMMDD HH:MM' to
        filter the coordinates.

    Returns:
        pandas.DataFrame: DataFrame containing columns [time, Xgsm, Ygsm,
        Zgsm],
                          where Xgsm, Ygsm, Zgsm are the GSM coordinates in
                          RE units.
                          The 'time' column contains datetime objects.
    """

    spc_coords = nc.Dataset(spc_coords_file)
    spcCoords_time = goes_epoch_to_datetime(
        spc_coords['time'][:]).to_pydatetime().tolist()
    tickz = spt.Ticktock(spcCoords_time, 'UTC')

    x, y, z = spc_coords['gse_xyz'][:, 0], spc_coords['gse_xyz'][:, 1], \
        spc_coords['gse_xyz'][:, 2]
    pos_gse = np.column_stack((x, y, z))
    pos_gse_coords = spcoords.Coords(pos_gse, 'GSE', 'car', ticks=tickz)

    # pos_x_gse, pos_y_gse, pos_z_gse = pos_gse_coords.x[:],
    # pos_gse_coords.y[:], pos_gse_coords.z[:]
    # spc_coords_df = pd.DataFrame({'time': spcCoords_time, 'X': pos_x_gse,
    # 'Y': pos_y_gse, 'Z': pos_z_gse})
    # Filter coordinates based on hour
    # spc_coords_df['hour'] = spc_coords_df['time'].apply(lambda x: x.hour)
    # filtered_coords_df = spc_coords_df[(spc_coords_df['hour'] ==
    # hour)].drop(columns=['hour'])

    spc_coords_df = pd.DataFrame(
        {'time': spcCoords_time, 'X': pos_gse_coords.x, 'Y': pos_gse_coords.y,
         'Z': pos_gse_coords.z})
    filtered_coords_df = spc_coords_df[spc_coords_df['time'] == timestamp]
    filtered_coords_df = filtered_coords_df.drop(columns=['time'])

    return filtered_coords_df


def transform_longitude_to_GSE(longitude, utc_time, is_west=False):
    """
    Transform the longitude of a geostationary satellite into GSE coordinates.

    Parameters
    ----------
    longitude (float): Longitude of the satellite in degrees. Positive for
    East, negative for West.
    utc_time (str): UTC time in the format 'HH:MM'.
    is_west (bool): Set to True if the longitude is provided in degrees West.

    Returns
    -------
    dict: GSE coordinates of the satellite in kilometers.
    """

    # Convert West longitude to East longitude if needed
    if is_west:
        longitude = 360 - longitude

    # Convert UTC time to angle
    utc_hour, utc_minute = map(int, utc_time.split(':'))
    total_utc_hours = utc_hour + utc_minute / 60
    earth_rotation_degrees_per_hour = 360 / 24
    earth_rotation_angle = total_utc_hours * earth_rotation_degrees_per_hour

    # Satellite fixed position relative to Greenwich Meridian
    fixed_longitude = longitude - earth_rotation_angle
    fixed_longitude_rad = np.radians(fixed_longitude)

    # Calculate GSE coordinates in Earth Radii (RE)
    x_re = np.cos(fixed_longitude_rad) * GEOSTAT
    y_re = np.sin(fixed_longitude_rad) * GEOSTAT
    z_re = 0  # Geostationary satellite, so Z coordinate is 0

    # Convert coordinates from RE to km
    x_km = x_re * RE_EARTH
    y_km = y_re * RE_EARTH
    z_km = z_re * RE_EARTH  # remains 0

    return {'X': x_km, 'Y': y_km, 'Z': z_km}


def process_sat_data_inputs(args):
    """
    Process the satellite data files based on the provided command-line
    arguments.
    Calculate GK2A position based on the longitude difference from G18.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary containing satellite data with satellite names as
        keys.
    """
    satellites_data = {}

    # Process data for other satellites if files are provided
    satellites = {
        'g16': args.g16,
        'g17': args.g17,
        'g18': args.g18
    }

    for satellite_name, satellite_file in satellites.items():
        if satellite_file:
            satellite_data = convert_and_filter_gse_by_timestamp(
                satellite_file, args.timestamp)
            satellites_data[satellite_name] = satellite_data
        else:
            print(f"No file provided for {satellite_name}.")

    if args.gk2a:
        # Extract G18's GSE coordinates from satellites_data
        g18_gse_coords = satellites_data.get('g18')

        if g18_gse_coords is not None:
            # Calculate the longitudinal difference in degrees
            longitudinal_difference = GK2A_LONG - (
                    360 - G18_LONG)  # Convert G18's longitude to degrees
            # East and add to GK2A's

            # Convert the longitudinal difference to radians
            longitudinal_difference_rad = np.radians(longitudinal_difference)

            # Assuming G18's position is the reference, calculate GK2A's GSE
            # position
            gk2a_gse_coords = {
                'X': g18_gse_coords['X'] * np.cos(
                    longitudinal_difference_rad) - g18_gse_coords[
                         'Y'] * np.sin(longitudinal_difference_rad),
                'Y': g18_gse_coords['X'] * np.sin(
                    longitudinal_difference_rad) + g18_gse_coords[
                         'Y'] * np.cos(longitudinal_difference_rad),
                'Z': g18_gse_coords['Z']  # Assuming Z remains the same
            }
            satellites_data['gk2a'] = gk2a_gse_coords
        else:
            print(
                "G18 data is required for calculating GK2A's position but is "
                "not available.")

    ic(satellites_data)
    return satellites_data


# def transform_spacecraft_data(satellites_data, gk2a_gse_coords=None):
#     """
#     Transform spacecraft data into a dictionary format for plotting,
#     replacing GK2A coordinates with calculated GSE coordinates.
#
#     Parameters:
#         satellites_data (dict): Dictionary containing the stacked
#         spacecraft data.
#         gk2a_gse_coords (dict): Calculated GSE coordinates for GK2A.
#
#     Returns:
#         dict: Dictionary with satellite names as keys and coordinate data
#         as values.
#     """
#     transformed_dict = {}
#
#     for satellite, data in satellites_data.items():
#         # Replace GK2A data with calculated GSE coordinates
#         if satellite == 'gk2a':
#             coords = {'X': gk2a_gse_coords['X'], 'Y': gk2a_gse_coords[
#             'Y'], 'Z': gk2a_gse_coords['Z']}
#         else:
#             # Process other satellite data
#             if isinstance(data, pd.DataFrame):
#                 data = data.to_numpy()
#
#             if data.shape[1] >= 3:
#                 coords = {'X': data[:, 0], 'Y': data[:, 1], 'Z': data[:, 2]}
#             else:
#                 raise ValueError(
#                     f"Data for satellite {satellite} does not have enough
#                     columns")
#
#         transformed_dict[satellite] = coords
#
#     return transformed_dict
def transform_spacecraft_data(satellites_data):
    """
    Transform spacecraft data into a dictionary format for plotting.

    Parameters:
        satellites_data (dict): Dictionary containing the spacecraft data.

    Returns:
        dict: Dictionary with satellite names as keys and coordinate data as
        values.
    """
    transformed_dict = {}

    for satellite, data in satellites_data.items():
        if isinstance(data, pd.DataFrame):
            # Extract the first row as it's assumed there's only one row per
            # satellite for the relevant timestamp
            coords = data.iloc[0].to_dict()
        elif isinstance(data, dict):
            # Assuming the dictionary contains pandas Series objects for X,
            # Y, Z
            coords = {coord: data[coord].iloc[0] for coord in ['X', 'Y', 'Z']}
        else:
            raise ValueError(
                f"Data for satellite {satellite} does not have the correct "
                f"format")

        transformed_dict[satellite] = coords

    return transformed_dict


def main():
    args = parse_arguments()

    timestamp_for_OMNI_title = args.timestamp
    timestampinHHMM = str(timestamp_for_OMNI_title[9:14])

    # get gse coords from gk2a longitude
    # gk2a_gse_coords_from_long = transform_longitude_to_GSE(GK2A_LONG,
    # timestampinHHMM)
    # gk2a_gse_coords_from_long = True

    satellites_data = process_sat_data_inputs(args)

    transformed_dict = transform_spacecraft_data(satellites_data)
    ic(transformed_dict)

    imf_bz, solar_wind_pressure = get_omni_values()

    # imf_bz, solar_wind_pressure = -13.75, 6.86
    plot_spacecraft_positions_with_earth_and_magnetopause(transformed_dict,
                                                          solar_wind_pressure,
                                                          imf_bz,
                                                          timestamp_for_OMNI_title)


if __name__ == "__main__":
    main()
