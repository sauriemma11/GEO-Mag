import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation
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

def calculate_magnetic_inclination_angle(bx, by, bz):
    return np.arctan2(bx, np.sqrt(by**2 + bz**2))

def convert_to_VDH(bxbybz, ut):
    print(ut, bxbybz)
    return None

def stack_from_data(sc_data):
    bx, by, bz = sc_data[:,0], sc_data[:,1], sc_data[:,2]
    b_gse_stacked = np.column_stack((bx, by, bz))
    return b_gse_stacked

def gse_to_geo(b_gse_stacked, time):
    # Need to create ticks from spacepy.time for unit conversion:
    tickz = spt.Ticktock(time, 'UTC')

    # Create GSE Coords object
    b_gse_coords = spc.Coords(b_gse_stacked, 'GSE', 'car', ticks=tickz)

    # Perform the transformation to geo
    geo_coords = b_gse_coords.convert('GEO', 'car')

    geo_coords_sph = b_gse_coords.convert('GEO', 'sph')  # Convert to spherical coordinates

    # Extract GEO longitude
    geo_longitude = geo_coords_sph.long

    return geo_coords, geo_longitude

def geo_to_rhenp(geo_long, geo_cart_coords, backward=False):
    # Calculate the Hapgood matrix based on geo_lon
    mat = hapgood_matrix(geo_long, 2)

    # If backward=True, transpose the matrix
    if backward:
        mat = np.transpose(mat)

    # Apply the transformation to geo_cart_coords using matrix multiplication (dot prod)
    result = np.dot(geo_cart_coords, mat)

    return result

def hapgood_matrix(theta, axis):
    """ Implementation of ATBD for GOES-R MAG Alternate Coordinate Systems.

        Hertitage:
            Adapted from Loto'aniu C++ implementation (grtransform.cpp::Grtransform::hapgood_matrix(...)).

            Original source: https://github.com/CIRES-STP/goesr_l2_mag_algs/blob/7d5155f3b98cd8e9503c877b423e54f36ede8c25/multi_alg_dependencies/src/python/common/goesr/goes_coordinates.py

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
            Adapted from Loto'aniu C++ implementation (Grtransform::mat_T_rhenp_vdh(...)).

    :param dt:      Scalar or Numpy array of date-times.
    :param geo_lat: Geocentric latitude. [Scalar or Numpy array of floats.]
    :param geo_lon: Geocentric longitude for determining Hapgood matrix. [Scalar or Numpy array of floats.]
    :param rhenp:   Right-handed ENP (intermediate coordinate system).
    :param mats:    Rotation (Hapgood) matrices (optional, default None, meaning they will be computed)
    :return:        Numpy array of VDH cartesian coordinates with dimension Nx3 (input units).
    """
    logger.info('STARTING: RH-ENP to VDH for %d coordinates.' % (len(dt)))

    # Pre-conditions
    n_points = len(dt)
    assert (np.shape(rhenp) == (n_points, 3))
    assert (np.shape(geo_lat) == (n_points,))
    assert (np.shape(geo_lon) == (n_points,))

    output = np.full((n_points, 3), np.nan, dtype=np.float)

    if mats is None:
        mats = rhenp_to_vdh_mats(dt, geo_lat, geo_lon)
    else:
        assert (np.shape(mats) == (n_points, 3, 3))
        logger.info('RH-ENP to VDH rotation matrices passed as optional input')

    # Each time step
    for i in np.arange(n_points):
        # Project the input vector(s) onto VDH:
        output[i, :] = np.dot(mats[i, :, :], rhenp[i, :])

    return output


def plot_magnetic_inclination_over_time(goes_time, goes_data, gk2a_data, date_str):
    # Calculate θ for GOES and GK2A data
    goes_theta = calculate_magnetic_inclination_angle(goes_data[:, 0], goes_data[:, 1], goes_data[:, 2])
    gk2a_theta = calculate_magnetic_inclination_angle(gk2a_data[:, 0], gk2a_data[:, 1], gk2a_data[:, 2])

    # Create plots for θ over time
    fig, (ax1) = plt.subplots()

    # GOES, red
    # SOSMAG, blue

    ax1.plot(goes_time, np.degrees(goes_theta), label='GOES', color='red')
    ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ) - {date_str}')
    ax1.set_ylabel('θ [degrees]')


    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pickle file paths:
    goes_pickle_path = 'Z:/Data/GOES18/model_outs/20221217/modout_20221217.pickle'
    gk2a_pickle_path = 'Z:/Data/GK2A/model_outputs/20221217/sosmag_modout_2022-12-17.pickle'

    # Load GOES and GK2A mag data
    goes_data = load_pickle_file(goes_pickle_path)['sat']
    gk2a_data = load_pickle_file(gk2a_pickle_path)['sat_gse']

    # Load the time data
    goes_time = load_pickle_file(goes_pickle_path)['time_min']
    gk2a_time = load_pickle_file(gk2a_pickle_path)['time_min']

    # For plot title, mainly
    date_str = '2022-12-17'

    # plot_magnetic_inclination_over_time(goes_time, goes_data, gk2a_data, date_str)

    geo_coords, geo_long = gse_to_geo(stack_from_data(goes_data), goes_time)

