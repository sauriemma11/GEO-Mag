import numpy as np
from netCDF4 import Dataset
import pandas as pd
import datetime as dtm
from datetime import timedelta
from icecream import ic
from hapiclient import hapi
import matplotlib.pyplot as plt
import json


def fetch_ace_data_from_hapi(server, dataset, start, end):
    """Fetch data for ACE spacecraft using the HAPI server."""
    data, meta = hapi(server, dataset, '', start, end)

    return data


def get_params_to_propagate(config, spacecraft):
    # Check if 'params_to_propagate' is defined in the config
    if 'params_to_propagate' in config:
        return config['params_to_propagate']
    else:
        if spacecraft == 'DSCOVR':
            return ['bz_gsm', 'proton_speed', 'proton_density']
        elif spacecraft == 'ACE':
            return ['V_GSM', 'Np']
        else:
            raise ValueError(f"Unknown spacecraft: {spacecraft}")


def process_ace_timedata(data):
    time_data = data['Time']
    # 64 sec cadence for SW and 16 sec for mag/pos
    time_data = [dtm.datetime.strptime(t.decode('utf-8'), '%Y-%m-%dT%H:%M:%S.%fZ') for t in time_data]
    return time_data


def propagate_parameters(config_path=None):
    """
    Main function for propagating the parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    spacecraft = config.get('spacecraft', '').upper()
    params = get_params_to_propagate(config, spacecraft)

    if spacecraft == 'DSCOVR':
        sw_fill_value = -99999
        pos_fill_value = -99999.0

        # 1 sec cadence
        files = config.get('files', {})
        mag_data = Dataset(files['mag_file'])
        sw_data = Dataset(files['sw_file'])
        pos_data = Dataset(files['pos_file'])

        def unix_ms_to_datetime(unix_ms):
            return np.array([dtm.datetime(1970, 1, 1) + timedelta(milliseconds=x) for x in unix_ms])

        mag_time = unix_ms_to_datetime(np.array(mag_data.variables['time'][:]))
        sw_time = unix_ms_to_datetime(np.array(sw_data.variables['time'][:]))
        pos_time = unix_ms_to_datetime(np.array(pos_data.variables['time'][:]))

        # Velocity and position data for propagation calculation
        vx_gsm_data = np.array(sw_data.variables['proton_vx_gsm'][:]).astype(float)
        posx_gsm_data = np.array(pos_data.variables['sat_x_gsm'][:]).astype(float)

    elif spacecraft == 'ACE':
        sw_fill_value = -1e+31
        pos_fill_value = -99999.0

        # 64 cadence data
        start = config['time_range']['start']
        end = config['time_range']['end']
        mfi_data = fetch_ace_data_from_hapi('https://cdaweb.gsfc.nasa.gov/hapi', 'AC_H0_MFI', start, end)
        sw_data = fetch_ace_data_from_hapi('https://cdaweb.gsfc.nasa.gov/hapi', 'AC_H0_SWE', start, end)

        sw_time = process_ace_timedata(sw_data)  # 64 sec
        mag_time = process_ace_timedata(mfi_data)  # 16 sec
        pos_time = mag_time  # 16 sec

        # Velocity and position data for propagation calculation
        vx_gsm_data = np.array(sw_data['V_GSM'][:, 0]).astype(float)
        posx_gsm_data = np.array(mfi_data['SC_pos_GSM'][:, 0]).astype(float)


    else:
        raise ValueError("Unsupported spacecraft")

    # Initialize propagated data dictionary dynamically
    # params = config.get('params_to_propagate', [])
    propagated_data = {param: [] for param in params}
    propagated_data['time'] = []

    posx_gsm_data[posx_gsm_data == pos_fill_value] = np.nan
    vx_gsm_data[vx_gsm_data <= sw_fill_value] = np.nan

    # Propagate each parameter
    for param in params:
        data = None
        if spacecraft == 'DSCOVR':
            if param in mag_data.variables:
                data = np.array(mag_data.variables[param][:]).astype(float)
                time = mag_time
            elif param in sw_data.variables:
                data = np.array(sw_data.variables[param][:]).astype(float)
                time = sw_time
            else:
                continue
        elif spacecraft == 'ACE':
            if param in sw_data.dtype.names:
                data = np.array(sw_data[param]).astype(float)
                time = sw_time
            elif param in mfi_data.dtype.names:
                data = np.array(mfi_data[param]).astype(float)
                time = mag_time
            else:
                continue

        data[data <= -9999] = np.nan

        # TODO: interp ACE values??????
        pos_interp = np.interp(np.arange(len(time)), np.arange(len(pos_time)), posx_gsm_data)

        # Calculate propagation time
        propagation_time = pos_interp / np.abs(vx_gsm_data)

        # Convert to DatetimeIndex for addition
        time_index = pd.DatetimeIndex(time)
        prop_indices = time_index + pd.to_timedelta(propagation_time, unit='s')

        propagated_data[param].extend(data)  # Store propagated data
        if 'time' not in propagated_data or not propagated_data['time']:
            propagated_data['time'].extend(prop_indices)

    return propagated_data


if __name__ == '__main__':
    propagated_data = propagate_parameters('../mploc_config.JSON')
