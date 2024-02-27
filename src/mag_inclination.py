from icecream import ic
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import pandas as pd
import datetime as dtm
import os
from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation
from utils import find_data_errors, fix_data_error_with_nan
from cdasws.datarepresentation import DataRepresentation as dr

cdas = CdasWs()
if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"
from spacepy import pycdf
import spacepy.coordinates as spc
import spacepy.time as spt


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def calculate_magnetic_inclination_angle_GSE(bx, by, bz):
    return np.arctan2(bx, np.sqrt(by ** 2 + bz ** 2))


def calculate_magnetic_inclination_angle_VDH(bV, bD, bH):
    return np.arctan2(bH, np.sqrt(bV ** 2 + bD ** 2))


def gse_to_vdh(gse_data, time):
    # Convert GSE to GEO
    geo_coords, geo_lat, geo_long = gse_to_geo(gse_data, time)

    # Convert GEO to RHENP
    rhenp_coords = geo_to_rhenp(geo_long, geo_coords)

    # Convert RHENP to VDH
    vdh_coords = rhenp_to_vdh(time, geo_lat, geo_long, rhenp_coords)

    return vdh_coords


def stack_from_data(sc_data):
    bx, by, bz = sc_data[:, 0], sc_data[:, 1], sc_data[:, 2]
    b_gse_stacked = np.column_stack((bx, by, bz))
    return b_gse_stacked


def gse_to_geo(b_gse_stacked, time):
    time = time.to_pydatetime()

    # Need to create ticks from spacepy.time for unit conversion:
    tickz = spt.Ticktock(time, 'UTC')

    # Create GSE Coords object
    b_gse_coords = spc.Coords(b_gse_stacked, 'GSE', 'car', ticks=tickz)

    # Perform the transformation to geo
    geo_coords = b_gse_coords.convert('GEO', 'car')

    geo_coords_sph = b_gse_coords.convert('GEO',
                                          'sph')  # Convert to spherical
    # coordinates

    # Extract GEO lat long
    geo_latitude, geo_longitude = geo_coords_sph.lati, geo_coords_sph.long

    return geo_coords, geo_latitude, geo_longitude


def geo_to_rhenp(geo_long, geo_coords, backward=False):
    n_points = len(geo_long)
    result = np.empty((n_points, 3))

    for i in range(n_points):
        mat = hapgood_matrix(geo_long[i], 2)

        if backward:
            mat = np.transpose(mat)

        geo_cart_coords = geo_coords.data[i,
                          :]  # Extract coordinates for a single time point
        result[i, :] = np.dot(geo_cart_coords, mat)

    return result


def hapgood_matrix(theta, axis):
    """ Implementation of ATBD for GOES-R MAG Alternate Coordinate Systems.

        Hertitage:
            Adapted from Loto'aniu C++ implementation (
            grtransform.cpp::Grtransform::hapgood_matrix(...)).

            Original source: https://github.com/CIRES-STP/goesr_l2_mag_algs
            /blob/7d5155f3b98cd8e9503c877b423e54f36ede8c25
            /multi_alg_dependencies/src/python/common/goesr/goes_coordinates.py

        :param theta: degrees to rotate
        :param axis: axis to rotate: 0, 1 or 2
        :return: Hapgood matrix
        """
    assert np.isscalar(theta) and np.isscalar(axis)

    sin_theta = np.sin(np.radians(theta))
    cos_theta = np.cos(np.radians(theta))

    t1, t2 = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)

    # initialize rotation matrix
    mat = np.zeros((3, 3))

    # Determine matrix diagonal
    #   1.put 1 in the Nth term, where N=1 if the rotation axis is X, etc
    #   2.put cos(zeta) in the other two terms
    mat[axis, axis] = 1.0
    mat[t1, t1] = cos_theta
    mat[t2, t2] = cos_theta

    # Locate the two off-diagonal terms in the same columns and rows as
    # the cos(zeta) terms - put sin(zeta) in the term above the diagonal
    # and -sin(zeta) in the term below,
    mat[t1, t2] = sin_theta
    mat[t2, t1] = -sin_theta

    # Return rotation matrix
    return mat


def rhenp_to_vdh(dt, geo_lat, geo_lon, rhenp, mats=None):
    """ The magnetic VDH coordinate definition and transformation algorithm
        follows that given by McPherron (1973) for the ATS-1.

        V: anti-earthward
        D: eastward
        H: northward along dipole

        Heritage:
            Implementation of ATBD for GOES-R MAG Alternate Coordinate Systems.
            Adapted from Loto'aniu C++ implementation (
            Grtransform::mat_T_rhenp_vdh(...)).

    :param dt:      Scalar or Numpy array of date-times.
    :param geo_lat: Geocentric latitude. [Scalar or Numpy array of floats.]
    :param geo_lon: Geocentric longitude for determining Hapgood matrix. [
    Scalar or Numpy array of floats.]
    :param rhenp:   Right-handed ENP (intermediate coordinate system).
    :param mats:    Rotation (Hapgood) matrices (optional, default None,
    meaning they will be computed)
    :return:        Numpy array of VDH cartesian coordinates with dimension
    Nx3 (input units).
    """

    # Pre-conditions
    n_points = len(dt)
    assert (np.shape(rhenp) == (n_points, 3))
    assert (np.shape(geo_lat) == (n_points,))
    assert (np.shape(geo_lon) == (n_points,))

    output = np.full((n_points, 3), np.nan, dtype=np.float)

    mats = rhenp_to_vdh_mats(dt, geo_lat, geo_lon)

    # Each time step
    for i in np.arange(n_points):
        # Project the input vector(s) onto VDH:
        output[i, :] = np.dot(mats[i, :, :], rhenp[i, :])

    return output


def rhenp_to_vdh_mats(dt, geo_lat, geo_lon):
    """ Calculate the rotation (Hapgood) matrices for RHENP to VDH
    coordinate transform.

    :param dt:      Scalar or Numpy array of date-times.
    :param geo_lat: Geocentric latitude. [Scalar or Numpy array of floats.]
    :param geo_lon: Geocentric longitude for determining Hapgood matrix. [
    Scalar or Numpy array of floats.]
    :return: 3D Numpy array of shape Nx3x3 representing each rotation matrix
    """

    mats = np.empty((len(dt), 3, 3))

    for i, (lat, lon) in enumerate(zip(geo_lat, geo_lon)):
        mat = np.zeros((3, 3))

        dip_lat, dip_lon = dipole_12_mag_lat_lon(dt[i])

        u = np.array([
            np.cos(np.radians(dip_lat)) * np.cos(np.radians(dip_lon)),
            np.cos(np.radians(dip_lat)) * np.sin(np.radians(dip_lon)),
            np.sin(np.radians(dip_lat))
        ])

        Rg = np.array([np.cos(np.radians(lat)), 0.0, np.sin(np.radians(lat))])

        mat_tmp = hapgood_matrix(lon, 2)

        H = np.dot(mat_tmp, u)

        Q = np.sqrt((H[1] * Rg[2] - Rg[1] * H[2]) ** 2 + (
                H[2] * Rg[0] - Rg[2] * H[0]) ** 2 + (
                            H[0] * Rg[1] - Rg[0] * H[1]) ** 2)

        tmp_vec1 = np.cross(H, Rg)

        D = tmp_vec1 / Q

        V = np.cross(D, H)

        mat[0, :] = V
        mat[1, :] = D
        mat[2, :] = H

        mats[i] = mat

    return mats


def dipole_12_mag_lat_lon(dt):
    """
    Calculates the IGRF12 Dipole latitude and longitude in Geocentric
    coordinates for a scalar.

    Heritage:
        Adapted from Loto'aniu dipole.cpp.

    :param dt: single Datetime object.
    :return: [ dipole_lat, dipole_lon ]    # degrees
    """

    assert len(np.shape([dt])) == 1, "Expected a scalar time value"

    ''' IGRF Constants: from 
    https://www.ngdc.noaa.gov/IAGA/vmod/igrf12coeffs.txt'''
    dip_epochs = np.array(
        [1900, 1905, 1910, 1915, 1920, 1925, 1930, 1935, 1940, 1945, 1950,
         1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005,
         2010, 2015, 2020])
    g01 = np.array(
        [-31543, -31464, -31354, -31212, -31060, -30926, -30805, -30715,
         -30654, -30594, -30554, -30500, -30421, -30334, -30220, -30100,
         -29992, -29873, -29775, -29692, -29619.4, -29554.63, -29496.57,
         -29441.46, -29404.8])
    g11 = np.array(
        [-2298, -2298, -2297, -2306, -2317, -2318, -2316, -2306, -2292, -2285,
         -2250, -2215, -2169, -2119, -2068, -2013, -1956, -1905, -1848, -1784,
         -1728.2, -1669.05, -1586.42, -1501.77, -1450.9])
    h11 = np.array(
        [5922, 5909, 5898, 5875, 5845, 5817, 5808, 5812, 5821, 5810, 5815,
         5820, 5791, 5776, 5737, 5675, 5604, 5500, 5406, 5306, 5186.1, 5077.99,
         4944.26, 4795.99, 4652.5])

    assert len(dip_epochs) == len(g01) == len(g11) == len(
        h11), "Coefficient arrays must all have same length"

    ''' Interpolate IGRF coefficients linearly '''
    # Nearest IGRF epoch <= Date
    i_epoch = np.where(dip_epochs <= dt.year)[0][-1]
    i_epoch_next = i_epoch + 1
    if (dip_epochs[i_epoch] == dip_epochs[-1]):
        i_epoch = len(dip_epochs) - 1
        i_epoch_next = i_epoch

    frac_year = year_fraction(dt)
    g01 = g01[i_epoch] + frac_year * (g01[i_epoch_next] - g01[i_epoch])
    g11 = g11[i_epoch] + frac_year * (g11[i_epoch_next] - g11[i_epoch])
    h11 = h11[i_epoch] + frac_year * (h11[i_epoch_next] - h11[i_epoch])

    ''' IGRF Northern pole geocentric Latitude and Longitude '''

    """ Longitude """
    dip_lon = np.degrees(np.arctan2(h11, g11) + np.pi)

    """ Latitude """
    rad_dip_lon = np.radians(dip_lon)
    dip_lat = np.degrees(
        np.pi / 2 - np.arctan(
            (g11 * np.cos(rad_dip_lon) + h11 * np.sin(rad_dip_lon)) / g01
        )
    )

    return [dip_lat, dip_lon]


def year_fraction(dt):
    '''
    Converts Datetime to Year fraction [0,1).

    :param dt: Datetime (scalar).
    :return: Year fraction (scalar) [0,1).
    '''
    import calendar

    days_in_year = 366. if calendar.isleap(dt.year) else 365.

    # This DOY is a fractional day of year within [0,365) or [0,366) (non
    # inclusive).
    day_of_year = (-1. + int(
        dt.strftime('%j'))) + dt.hour / 24. + dt.minute / 1440. \
                  + (dt.second + 1e-6 * dt.microsecond) / 86400.

    year_frac = day_of_year / days_in_year  # Between 0 and 1

    return year_frac


def fix_nan_for_goes(data, nanvalue=-9998.0):
    data[data < nanvalue] = np.nan
    return data


def goes_epoch_to_datetime(timestamp):
    """
    Converts goes epoch time from .cda into pandas datetime timestamp

    :param timestamp: from .cda file
    :return: pandas datetime timestamp
    """
    epoch = pd.to_datetime('2000-01-01 12:00:00')
    time_datetime = epoch + pd.to_timedelta(timestamp, unit='s')
    return time_datetime


def plot_magnetic_inclination_over_time(goes_time, goes_data, gk2a_data,
                                        date_str):
    # Calculate θ for GOES and GK2A data
    goes_theta = calculate_magnetic_inclination_angle_VDH(goes_data[:, 0],
                                                          goes_data[:, 1],
                                                          goes_data[:, 2])
    gk2a_theta = calculate_magnetic_inclination_angle_VDH(gk2a_data[:, 0],
                                                          gk2a_data[:, 1],
                                                          gk2a_data[:, 2])

    # Create plots for θ over time
    fig, (ax1) = plt.subplots()

    # GOES, red
    # SOSMAG, blue

    ax1.plot(goes_time, np.degrees(goes_theta), label='GOES', color='red',
             linewidth=1)
    ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue',
             linewidth=1)

    ax1.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax1.set_ylabel('θ [degrees]')
    ax1.set_ylim(0, 90)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

    plt.tight_layout()
    plt.show()

def plot_magnetic_inclination_over_time_3sc(goes_time, date_str,
                                            datasets):
    color_map = {
        'G17': 'red',
        'G18': 'orange',
        'GK2A': 'blue',
        'G16': 'green'
    }

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot each dataset
    for satellite, data in datasets.items():
        if satellite in color_map and data is not None:
            # Assume a function calculate_magnetic_inclination_angle exists
            theta = calculate_magnetic_inclination_angle_VDH(data[:, 0],
                                                             data[:, 1],
                                                             data[:, 2])
            ax.plot(goes_time, np.degrees(theta), label=satellite,
                    color=color_map[satellite], linewidth=1)

    ax.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax.set_ylabel('θ [degrees]')
    ax.set_ylim(-45, 90)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.legend()

    plt.tight_layout()
    plt.show()



def plot_BGSE_fromdata(spacecraftdata, whatspacecraft):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.set_title(f'{whatspacecraft} B_GSE')
    ax1.plot(spacecraftdata[:, 0], label='X')
    ax2.plot(spacecraftdata[:, 1], label='Y')
    ax3.plot(spacecraftdata[:, 2], label='Z')
    plt.tight_layout()
    ax1.legend()
    plt.show()


def plot_BGSE_fromdata_ontop(timedataset, date_str, spacecraft_data_dict):
    """
    Plots the B field from multiple spacecraft data on top of each other.

    Parameters:
    timedataset (List[datetime]): The time data for the x-axis.
    spacecraft_data_dict (dict): A dictionary where the key is the
    spacecraft name and the value is its corresponding data array.
    date_str (str, optional): The date string for the title of the plot.

    Returns:
    None
    """

    # Define the color mapping for each spacecraft
    color_map = {
        'G17': 'red',
        'G18': 'orange',
        'GK2A': 'blue',
        'G16': 'green'
    }

    # Create subplots
    fig, axs = plt.subplots(4, 1)

    # Set common plot properties
    for ax in axs[:-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(''))
        ax.tick_params(axis='x', which='both', length=6, labelbottom=False)

    # Set unique properties and plot each spacecraft
    for idx, (component, ax) in enumerate(zip(['X', 'Y', 'Z'], axs[:-1])):
        for spacecraft, data in spacecraft_data_dict.items():
            ax.plot(timedataset, data[:, idx], label=spacecraft,
                    color=color_map[spacecraft], linewidth=1)
            ax.set_ylabel(f'$B_{component.lower()}$ [nT]')
            # Only show the legend on the first plot (X component)
        if idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Plot sym-h for 4th subplot:
    # get symh data via cdasWs omni dataset
    data = cdas.get_data('OMNI_HRO_1MIN', 'SYM_H', f'{date_str}T00:00:00Z',
                         f'{date_str}T23:59:00Z',
                         dataRepresentation=dr.XARRAY)[
        1]
    sym_h = data.SYM_H.values

    # Plot the SYM-H data on the last subplot
    axs[-1].plot(timedataset, sym_h, label='SYM-H', linewidth=1)
    axs[-1].set_ylabel('SYM-H [nT]')
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    axs[-1].tick_params(axis='x', which='both')
    axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))

    axs[0].set_title(f'B field (GSE) - {date_str}')

    plt.subplots_adjust(hspace=0.075, right=0.8)
    plt.tight_layout()
    plt.show()


##################################

g16_dataset = nc.Dataset(
    'C:/Users/sarah.auriemma/Desktop/Data_new/g16/mag_1m/2019_05/dn_magn-l2'
    '-avg1m_g16_d20190513_v2-0-2.nc')
goes17coloc_dataset = nc.Dataset(
    'C:/Users/sarah.auriemma/Desktop/Data_new/g17/mag_1m/2019_05/dn_magn-l2'
    '-avg1m_g17_d20190513_v2-0-2.nc')
gk2a_dataset = nc.Dataset('Z:/Data/GK2A/SOSMAG_20190513_b_gse.nc')

goes_time_fromnc = goes_epoch_to_datetime(goes17coloc_dataset['time'][:])

# goes18_bgse_stacked = stack_from_data(goes18_dataset['b_gse'])
# goes18_bgse_stacked = fix_nan_for_goes(goes18_bgse_stacked)

goes17_bgse_stacked = stack_from_data(goes17coloc_dataset['b_gse'])
goes17_bgse_stacked = fix_nan_for_goes(goes17_bgse_stacked)

goes16_bgse_stacked = stack_from_data(g16_dataset['b_gse'])
goes16_bgse_stacked = fix_nan_for_goes(goes16_bgse_stacked)

gk2a_bgse_stacked = np.column_stack((gk2a_dataset['b_xgse'][:],
                                     gk2a_dataset['b_ygse'][:],
                                     gk2a_dataset['b_zgse'][:]))
date_str = dtm.datetime.strftime(goes_time_fromnc[0], '%Y-%m-%d')

# plot_BGSE_fromdata(goes17_bgse_stacked, 'goes17')
# plot_BGSE_fromdata(goes18_bgse_stacked, 'goes18')


# GOES17, red
# GOES18, orange
# SOSMAG, blue
# G16, green

spacecraft_data = {
    'G17': goes17_bgse_stacked,
    'G16': goes16_bgse_stacked,
    'GK2A': gk2a_bgse_stacked
}

plot_BGSE_fromdata_ontop(timedataset=goes_time_fromnc, date_str=date_str,
                         spacecraft_data_dict=spacecraft_data)

goes17_VDH = gse_to_vdh(goes17_bgse_stacked, goes_time_fromnc)
# goes18_VDH = gse_to_vdh(goes18_bgse_stacked, goes_time_fromnc)
gk2a_VDH = gse_to_vdh(gk2a_bgse_stacked, goes_time_fromnc)
goes16_VDH = gse_to_vdh(goes16_bgse_stacked, goes_time_fromnc)

VDH_Datasets = {'G17': goes17_VDH,
                'G16': goes16_VDH,
                'GK2A': gk2a_VDH}

plot_magnetic_inclination_over_time_3sc(goes_time_fromnc, date_str,
                                        VDH_Datasets)
