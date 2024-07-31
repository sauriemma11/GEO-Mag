import json
import logging
import datetime
from netCDF4 import num2date
import netCDF4 as nc
import numpy as np
from datetime import datetime
from icecream import ic
import os
from cdasws import CdasWs
from plotting.mploc_plotting import make_mpause_plots
from DSCOVR_prop.dscovr_propagation import propagate_parameters

if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"

cdas = CdasWs()

"""Constants"""
TIME_UNITS = "seconds since 2000-01-01 12:00:00"
GEOSTAT_re = 6.6  # geostationary orbit - Re
"""/Constants"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def convert_to_datetime(time_array, units):
    """
    Convert a numeric time array into an array of datetime objects.

    Parameters:
    - time_array (np.ndarray): An array of time values in numeric format.
    - units (str): A string describing the time units, according to the CF conventions,
                   e.g., "seconds since 2000-01-01 12:00:00".

    Returns:
    - np.ndarray: An array of datetime.datetime objects.
    """
    # Convert numeric time values to datetime objects
    cftime_array = num2date(time_array, units)

    # Convert cftime.datetime objects to standard datetime.datetime objects
    datetime_array = [
        datetime(cf.year, cf.month, cf.day, cf.hour, cf.minute, cf.second) for
        cf in cftime_array]

    return datetime_array


def get_omni_values(start_datetime, end_datetime):
    """
    Fetch solar wind values from the OMNI dataset for a specified datetime range.

    Parameters:
        start_datetime (datetime): Start datetime of the range.
        end_datetime (datetime): End datetime of the range.

    Returns:
        dict: A dictionary containing arrays for BZ_GSM, Pressure, and Speed.
    """
    # Formatting the start and end times to string
    start_time_str = start_datetime.strftime('%Y-%m-%dT%H:%M:00Z')
    end_time_str = end_datetime.strftime('%Y-%m-%dT%H:%M:00Z')

    cdas = CdasWs()

    # ic(cdas.get_variable_names('OMNI_HRO_1MIN'))

    # Fetch data from the OMNI dataset
    data = cdas.get_data(
        'OMNI_HRO_1MIN',
        ['BZ_GSM', 'flow_speed', 'proton_density'],
        start_time_str,
        end_time_str)

    response_metadata, actual_data = data

    if 'BZ_GSM' in actual_data:
        actual_data['BZ_GSM'] = np.where(actual_data['BZ_GSM'] >= 9999, np.nan, actual_data['BZ_GSM'])
    # if 'Pressure' in actual_data:
    # actual_data['Pressure'] = np.where(actual_data['Pressure'] >= 99, np.nan, actual_data['Pressure'])
    if 'flow_speed' in actual_data:
        actual_data['flow_speed'] = np.where(actual_data['flow_speed'] >= 99999, np.nan, actual_data['flow_speed'])
    if 'proton_density' in actual_data:
        actual_data['proton_density'] = np.where(actual_data['proton_density'] >= 999, np.nan,
                                                 actual_data['proton_density'])

    return actual_data


def calculate_solar_wind_dynamic_pressure(sw_bow_data):
    """
    Calculate the solar wind dynamic pressure from density and flow speed data.

    Parameters:
        sw_bow_data (dict): A dictionary containing solar wind data arrays including
                            'Pressure' for density and 'flow_speed' for speed.

    Returns:
        np.ndarray: An array of solar wind dynamic pressure values.
    """
    # Extract density and speed from the dictionary
    sw_density = sw_bow_data.get('proton_density')
    sw_speed = sw_bow_data.get('flow_speed')

    # Constants for the dynamic pressure calculation
    factor = 2e-6

    # Handle potential fill values and perform calculation only on valid data
    valid_density = np.where(sw_density != 99.99, sw_density, np.nan)
    valid_speed = np.where(sw_speed != 99999.9, sw_speed, np.nan)

    # Calculate dynamic pressure only where neither density nor speed are fill values
    sw_pdyn = factor * valid_density * valid_speed ** 2
    # proton_dom_sw_pdyn = constants.proton_mass * valid_density * valid_speed ** 2

    return sw_pdyn


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    logger.info("Configuration loaded successfully.")
    return config


def read_nc_data(filepath):
    with nc.Dataset(filepath, 'r') as dataset:
        # Check available variables in the file
        available_vars = dataset.variables.keys()
        # ic(available_vars)
        data = {}


        # Check if it's a magnetic field file
        if 'b_gsm' in available_vars:
            data['b_gsm'] = dataset.variables['b_gsm'][:]
            data['time'] = dataset.variables['time'][:]
            data['orbit_llr_geo'] = dataset.variables['orbit_llr_geo'][:]
            # logger.info(f"Magnetic field data (b_gsm) read from {filepath}")

        if 'b_epn' in available_vars:
            data['b_epn'] = dataset.variables['b_epn'][:]
            # logger.info(f"Magnetic field data (b_epn) read from {filepath}")

        # Check if it's a particle moments file
        if 'EleMoments' in available_vars:
            data['EleMoments'] = dataset.variables['EleMoments'][:]
            # logger.info(f"Electron moments data (EleMoments) read from {filepath}")

        if 'IonMoments' in available_vars:
            data['IonMoments'] = dataset.variables['IonMoments'][:]
            # ic(dataset.variables['IonMoments'])

            # logger.info(f"Ion moments data (IonMoments) read from {filepath}")

    return data


def process_mag_data(mag_data):
    """
    Replace fill values in the magnetic field data arrays ('b_gsm', 'b_epn') with NaNs.

    Parameters:
        mag_data (dict): Dictionary containing magnetic field arrays.

    Returns:
        dict: Dictionary with processed data where fill values have been replaced by NaNs.
    """
    processed_data = {}
    for key in ['b_gsm', 'b_epn']:
        if key in mag_data:
            data = mag_data[key]
            # If the data array is masked, replace the fill value with NaN
            if np.ma.is_masked(data):
                data = np.where(data.mask, np.nan, data.data)  # Use data.data to access the underlying numpy array
            else:
                # Assume fill values are set to 1e+20, replace them with NaN
                data = np.where(data == 1e+20, np.nan, data)

            # Store the cleaned data
            processed_data[key] = data

    return processed_data


def extract_moment_data(moment_data, fill_value=-1e+31):
    """
    Extract density, T_parallel, and T_perp from the moment data arrays, replacing any fill values with NaNs.

    Parameters:
        moment_data (np.ndarray): Array containing moment data for either electrons or ions.
        fill_value (float): The fill value that needs to be replaced with NaN.

    Returns:
        tuple of np.ndarrays: Density, T_parallel, and T_perp values, with fill values replaced by NaNs.
    """
    if moment_data.shape[2] < 3:
        raise ValueError("Expected at least three moments in the data")

    # Apply mask and replace fill values
    if np.ma.is_masked(moment_data):
        moment_data = np.ma.filled(moment_data, np.nan)  # Replace masked values with NaN

    # Replace fill values with NaNs before extraction
    clean_data = np.where(moment_data == fill_value, np.nan, moment_data)

    density = clean_data[:, 0, 0]  # time, first energy range, density
    t_parallel = clean_data[:, 0, 1]  # time, first energy range, T_parallel
    t_perp = clean_data[:, 0, 2]  # time, first energy range, T_perp

    return density, t_parallel, t_perp


def calc_ratio(moments):
    """
    Currently both electron moments and ion moments are in the format:

    Electron density, T_parallel, T_perp, anisotropy in specified energy ranges"
    "Ion density, T_parallel, T_perp, anisotropy in specified energy ranges"

    So the same indicies can be used to compute the ratio (density/temperature),
    where temperature = (2T_perp + T_para)/3

    See Suvorova 2005: https://doi.org/10.1029/2003JA010079

    Assumes moments is an array with dimensions (time, energy_range, moments), uses the first energy range.

    Errors can happen when extreme low/high temp values or near-to/zero values.
    """
    energy_range_index = 0
    density = moments[:, energy_range_index, 0]  # units: [1/cm^3]
    t_perp = moments[:, energy_range_index, 2]  # units: [keV]
    t_para = moments[:, energy_range_index, 1]  # units: [keV]
    temperature = (2 * t_perp + t_para) / 3.0  # Suvorova 2005 eq (11)
    return density / temperature  # Suvorova 2005 eq (10)


def run_shue(sw_bz, sw_pdyn):
    """
    Calculate the Shue et al. (1998) model for magnetopause location.

    Parameters:
        sw_bz (np.array): Solar wind Bz component [nT].
        sw_pdyn (np.array): Solar wind dynamic pressure [nPa].

    Returns:
        tuple: (r_0, alpha)
    """
    GEOSTAT_re = 6.6  # Define or ensure this constant is correctly imported or set

    # Ensure inputs are numpy arrays
    sw_bz = np.array(sw_bz, dtype=float)
    sw_pdyn = np.array(sw_pdyn, dtype=float)

    # Check if the input arrays are of the same size and 1-dimensional
    if sw_bz.shape != sw_pdyn.shape or sw_bz.ndim != 1:
        raise ValueError("sw_bz and sw_pdyn must be 1D arrays of the same length")

    # Calculate Shue et al. parameters
    shue_r0 = (10.22 + 1.29 * np.tanh(0.184 * (sw_bz + 8.14))) * np.power(sw_pdyn, -1.0 / GEOSTAT_re)
    shue_alpha = (0.58 - 0.007 * sw_bz) * (1 + 0.024 * np.log(sw_pdyn))

    return shue_r0, shue_alpha


def process_satellite(config, satellite_key):
    results = {}
    magn_data = read_nc_data(config[f'{satellite_key}_magn_file'])
    mpsl_data = read_nc_data(config[f'{satellite_key}_mpsl_file'])

    # Convert time array to datetime
    datetime_values = convert_to_datetime(magn_data['time'], units=TIME_UNITS)
    results['start_datetime'] = datetime_values[0]
    results['end_datetime'] = datetime_values[-1]

    results['datetime_values'] = datetime_values  # Storing the entire datetime array

    # Process magnetic field data if available
    if 'b_gsm' in magn_data or 'b_epn' in magn_data:
        magnetic_field_results = process_mag_data(magn_data)
        results.update(magnetic_field_results)

    # Process moment data if available
    if 'IonMoments' in mpsl_data:
        i_density, i_t_parallel, i_t_perp = extract_moment_data(mpsl_data['IonMoments'])
        i_moments = np.stack((i_density, i_t_parallel, i_t_perp), axis=-1).reshape(-1, 1, 3)
        i_ratios = calc_ratio(i_moments)
        results['ion_ratios'] = i_ratios

    if 'EleMoments' in mpsl_data:
        e_density, e_t_parallel, e_t_perp = extract_moment_data(mpsl_data['EleMoments'])
        e_moments = np.stack((e_density, e_t_parallel, e_t_perp), axis=-1).reshape(-1, 1, 3)
        e_ratios = calc_ratio(e_moments)
        results['electron_ratios'] = e_ratios

    return results


def calculate_flags(shue_r0, ion_ratios, electron_ratios, b_epn):
    """
    Calculate various flags based on provided conditions.

    Parameters:
        shue_r0 (np.ndarray): Array of shue_r0 values.
        ion_ratios (np.ndarray): Array of ion density to temperature ratios.
        electron_ratios (np.ndarray): Array of electron density to temperature ratios.
        b_epn (np.ndarray): Array of magnetic field data in EPN coordinates.
        thresholds (dict): Dictionary containing threshold values for flags.

    Returns:
        dict: Dictionary with flag arrays.
    """
    flags = {}
    geostationary_threshold = GEOSTAT_re  # Re for geostationary orbit
    flags['flag_r0'] = (shue_r0 < geostationary_threshold).astype(int)

    ion_ratio_threshold = 30  # Default to 30 if not specified
    flags['flag_ions'] = (ion_ratios >= ion_ratio_threshold).astype(int)

    electron_ratio_threshold = 100  # Default to 100 if not specified
    flags['flag_electrons'] = (electron_ratios >= electron_ratio_threshold).astype(int)

    # Assuming b_epn is structured with Hp being 2nd component b_epn[:,1]
    flags['flag_b_field'] = (b_epn[:, 1] <= 0).astype(int)  # True if Hp <= 0

    return flags


def rename_propagated_data_keys(propagated_data):
    """
    Rename keys in the propagated data dictionary to match expected OMNI data field names.

    Parameters:
        propagated_data (dict): Dictionary with keys as parameter names and values as data lists.

    Returns:
        dict: A dictionary with renamed keys.
    """
    # Mapping of original keys to desired OMNI-compatible keys
    key_map = {
        'bz_gsm': 'BZ_GSM',
        'proton_speed': 'flow_speed',
        'proton_density': 'proton_density',
        'time': 'Epoch'
    }

    # Create a new dictionary with the renamed keys
    renamed_data = {}
    for old_key, new_key in key_map.items():
        if old_key in propagated_data:
            renamed_data[new_key] = propagated_data[old_key]

    return renamed_data


def main(config_path):
    config = load_config(config_path)
    results = {}
    satellite_keys = ['g16', 'g17', 'g18']  # List of possible satellites

    # Process available satellite data
    for key in satellite_keys:
        if f'{key}_magn_file' in config:
            results[key] = process_satellite(config, key)
            if not results[key]:
                logger.warning(f"No data available for {key.upper()}.")

    # Determine the common datetime range for OMNI data fetch
    all_dates = [res['datetime_values'] for res in results.values() if res]
    if not all_dates:
        logger.error("No satellite data available.")
        return

    start_datetime = max([dates[0] for dates in all_dates])
    end_datetime = min([dates[-1] for dates in all_dates])

    if config.get('use_dscovr_propagation', False):
        ic('Getting SW data via DSCOVR Propagation')
        propagated_data = propagate_parameters(config_path=config_path)
        sw_data = rename_propagated_data_keys(propagated_data)
        sw_data_via = 'DSCOVR'
    else:
        print("Getting SW data via OMNI")
        # Fetch OMNI data using the determined datetime range
        sw_data = get_omni_values(start_datetime, end_datetime)
        sw_data_via = 'OMNI_HRO'

    sw_dyn_p = calculate_solar_wind_dynamic_pressure(sw_data)
    shue_r0, shue_alpha = run_shue(sw_data['BZ_GSM'], sw_dyn_p)
    ic(np.nanmin(shue_r0))

    # Calculate flags and plot results for each satellite
    for key, res in results.items():
        if res:
            satellite_name = f"GOES-{key[1:].upper()}"  # Construct the satellite name dynamically
            flags = calculate_flags(shue_r0, res['ion_ratios'], res['electron_ratios'], res['b_epn'])
            make_mpause_plots(res, flags, sw_data, shue_r0, sw_dyn_p, satellite_name, sw_data_via)


if __name__ == '__main__':
    main('mploc_config.JSON')
