from coord_transform import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
NOTE - Color of spacecraft should be same across all plotting functions
GOES17 : red
GOES18 : orange
SOSMAG : blue
"""


def plot_magnetic_inclination_over_time_3sc(date_str, goes_time, goes17_data=None, goes18_data=None, gk2a_data=None, save_path=None):
    """
    Plot magnetic inclination angle (θ) for multiple spacecraft over time.
    ** Note: All 3 s/c are optional, but at least one must be provided.

    :param date_str: A string representing the date (e.g., "YYYY-MM-DD").
    :param goes_time: The timestamp data for the plotted time.
    :param goes17_data: Data for GOES-17 (optional).
    :param goes18_data: Data for GOES-18 (optional).
    :param gk2a_data: Data for SOSMAG (optional).
    :param save_path: The file path to save the generated plot (optional).

    Example:
    plot_magnetic_inclination_over_time_3sc(date_str, goes_time, goes17_data, goes18_data, gk2a_data, save_path)
    """
    fig, (ax1) = plt.subplots()

    if goes17_data is not None:
        goes17_theta = calculate_magnetic_inclination_angle_VDH(goes17_data[:, 0],goes17_data[:, 1],goes17_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes17_theta), label='G17', color='red')

    if goes18_data is not None:
        goes18_theta = calculate_magnetic_inclination_angle_VDH(goes18_data[:, 0],goes18_data[:, 1],goes18_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes18_theta), label='G18', color='orange')

    if gk2a_data is not None:
        gk2a_theta = calculate_magnetic_inclination_angle_VDH(gk2a_data[:, 0], gk2a_data[:, 1], gk2a_data[:, 2])
        ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax1.set_ylabel('θ [degrees]')
    ax1.set_ylim(0, 90)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.show()


def plot_BGSE_fromdata_ontop(spacecraftdata1=None, spacecraftdata2=None, whatspacecraft1=None, whatspacecraft2=None, whatspacecraft3=None, spacecraftdata3=None, date_str=None, save_path=None):
    """
    Plot the B_GSE data from multiple spacecraft on top of each other.
    ** Note: All 3 spacecraft are optional, but at least one must be provided.

    :param spacecraftdata1: Data for the first spacecraft to plot (optional).
    :param spacecraftdata2: Data for the second spacecraft to plot (optional).
    :param whatspacecraft1: A label for the first spacecraft (optional).
    :param whatspacecraft2: A label for the second spacecraft (optional).
    :param whatspacecraft3: A label for the third spacecraft (optional).
    :param spacecraftdata3: Data for the third spacecraft (optional).
    :param date_str: A string representing the date (e.g., "YYYY-MM-DD").
    :param save_path: The file path to save the generated plot (optional).

    Example:
    plot_BGSE_fromdata_ontop(spacecraftdata1, spacecraftdata2, whatspacecraft1, whatspacecraft2, whatspacecraft3, spacecraftdata3, date_str, save_path)
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    if date_str is not None:
        ax1.set_title(f'B_GSE, {date_str}')

    if spacecraftdata1 is not None and whatspacecraft1 is not None:
        ax1.plot(spacecraftdata1[:, 0], label=f'{whatspacecraft1} X', color='red')
        ax2.plot(spacecraftdata1[:, 1], label=f'{whatspacecraft1} Y', color='red')
        ax3.plot(spacecraftdata1[:, 2], label=f'{whatspacecraft1} Z', color='red')

    if spacecraftdata2 is not None and whatspacecraft2 is not None:
        ax1.plot(spacecraftdata2[:, 0], label=f'{whatspacecraft2} X', color='orange')
        ax2.plot(spacecraftdata2[:, 1], label=f'{whatspacecraft2} Y', color='orange')
        ax3.plot(spacecraftdata2[:, 2], label=f'{whatspacecraft2} Z', color='orange')

    if spacecraftdata3 is not None and whatspacecraft3 is not None:
        ax1.plot(spacecraftdata3[:, 0], label=f'{whatspacecraft3} X', color='blue')
        ax2.plot(spacecraftdata3[:, 1], label=f'{whatspacecraft3} Y', color='blue')
        ax3.plot(spacecraftdata3[:, 2], label=f'{whatspacecraft3} Z', color='blue')

    plt.tight_layout()
    ax1.legend()

    # TODO: Fix save file, it is not working currently.
    if save_path:
        save_file_as = f'BGSE_{date_str}'
        plt.savefig(f"{save_path}/{save_file_as}.png")
        print(f'fig saved as {save_file_as}')
        plt.show()
    else:
        plt.show()

