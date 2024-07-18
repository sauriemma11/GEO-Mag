import numpy as np
import json
from netCDF4 import Dataset
import pandas as pd
import datetime as dtm
from datetime import timedelta
from icecream import ic
import math
import os
import json


def propagate_parameters(config_path=None, params=None):
    if config_path is None:
        # Assumes the script is run standalone and config file is in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of the current script
        config_path = os.path.join(script_dir, 'dscovr_prop_config.JSON')

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract file paths from the configuration
    mag_file = config.get('mag_file', '')
    sw_file = config.get('sw_file', '')
    pos_file = config.get('pos_file', '')

    # Load data from files
    mag_data = Dataset(mag_file)
    sw_data = Dataset(sw_file)
    pos_data = Dataset(pos_file)

    def unix_ms_to_datetime(unix_ms):
        return np.array([dtm.datetime(1970, 1, 1) + timedelta(milliseconds=x) for x in unix_ms])

    # Initialize propagated data dictionary dynamically
    propagated_data = {param: [] for param in params}
    propagated_data['time'] = []

    # Time conversions
    mag_time = unix_ms_to_datetime(np.array(mag_data.variables['time'][:]))
    sw_time = unix_ms_to_datetime(np.array(sw_data.variables['time'][:]))
    pos_time = unix_ms_to_datetime(np.array(pos_data.variables['time'][:]))

    # Velocity and position data for propagation calculation
    vx_gsm_data = np.array(sw_data.variables['proton_vx_gsm'][:]).astype(float)
    posx_gsm_data = np.array(pos_data.variables['sat_x_gsm'][:]).astype(float)

    # Clean position and velocity data
    vx_gsm_data[vx_gsm_data <= -99999] = np.nan  # Adjust threshold as necessary
    posx_gsm_data[posx_gsm_data == -99999.0] = np.nan

    # Propagate each parameter
    for param in params:
        data = None
        if param in mag_data.variables:
            data = np.array(mag_data.variables[param][:]).astype(float)
            time = mag_time
        elif param in sw_data.variables:
            data = np.array(sw_data.variables[param][:]).astype(float)
            time = sw_time
        else:
            continue

        data[data <= -9999] = np.nan

        pos_interp = np.interp(np.arange(len(time)), np.arange(len(pos_time)), posx_gsm_data)
        propagation_time = pos_interp / np.abs(vx_gsm_data)
        prop_indices = pd.to_datetime(time + pd.to_timedelta(propagation_time, unit='s'))

        propagated_data[param].extend(data)  # Storing propagated data
        if 'time' not in propagated_data or not propagated_data['time']:
            propagated_data['time'].extend(prop_indices)

    return propagated_data


# Example usage
if __name__ == '__main__':
    params_to_propagate = ['bz_gsm', 'proton_speed', 'proton_density']
    propagated_data = propagate_parameters(params=params_to_propagate)

    # Available variables:
    # Dataset(magfile).variables.keys(): dict_keys(['time', 'sample_count', 'measurement_mode', 'measurement_range',
    # 'bt', 'bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm',
    # 'phi_gsm', ... ])
    # Dataset(swfile).variables.keys(): dict_keys(['time', 'sample_count', 'proton_vx_gse', 'proton_vy_gse',
    # 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density',
    # 'proton_temperature', 'alpha_vx_gse', 'alpha_vy_gse', 'alpha_vz_gse', 'alpha_vx_gsm', 'alpha_vy_gsm',
    # 'alpha_vz_gsm', 'alpha_speed', 'alpha_density', 'alpha_temperature', ... ])
    # Dataset(posfile).variables.keys(): dict_keys(['time', 'sat_x_gci', 'sat_y_gci', 'sat_z_gci', 'sat_vx_gci',
    # 'sat_vy_gci', 'sat_vz_gci', 'sat_x_gse', 'sat_y_gse', 'sat_z_gse', 'sat_vx_gse', 'sat_vy_gse', 'sat_vz_gse',
    # 'sat_x_gsm', 'sat_y_gsm', 'sat_z_gsm', 'sat_vx_gsm', 'sat_vy_gsm', 'sat_vz_gsm'])

#
# # Usage
# magfile = 'Z:/Data/DSCOVR/2024/05/oe_m1m_dscovr_s20240510000000_e20240510235959_p20240511034031_pub.nc'
# swfile = 'Z:/Data/DSCOVR/2024/05/oe_f1m_dscovr_s20240510000000_e20240510235959_p20240511034609_pub.nc'
# posfile = 'Z:/Data/DSCOVR/2024/05/oe_pop_dscovr_s20240510000000_e20240510235959_p20240511034641_pub.nc'
#
# ic(Dataset(magfile).variables.keys())
# ic(Dataset(swfile).variables.keys())
# ic(Dataset(posfile).variables.keys())
