from coord_transform import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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

    ax1.plot(goes_time, np.degrees(goes_theta), label='GOES', color='red')
    ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax1.set_ylabel('θ [degrees]')
    ax1.set_ylim(0, 90)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

    plt.tight_layout()
    plt.show()

def plot_magnetic_inclination_over_time_3sc(goes_time, goes17_data, goes18_data, gk2a_data,
                                        date_str):
    # Calculate θ for GOES and GK2A data
    goes17_theta = calculate_magnetic_inclination_angle_VDH(goes17_data[:, 0],
                                                          goes17_data[:, 1],
                                                          goes17_data[:, 2])
    goes18_theta = calculate_magnetic_inclination_angle_VDH(goes18_data[:, 0],
                                                          goes18_data[:, 1],
                                                          goes18_data[:, 2])
    gk2a_theta = calculate_magnetic_inclination_angle_VDH(gk2a_data[:, 0],
                                                          gk2a_data[:, 1],
                                                          gk2a_data[:, 2])

    # Create plots for θ over time
    fig, (ax1) = plt.subplots()

    # GOES17, red
    # GOES18, orange
    # SOSMAG, blue

    ax1.plot(goes_time, np.degrees(goes17_theta), label='G17', color='red')
    ax1.plot(goes_time, np.degrees(goes18_theta), label='G18', color='orange')
    ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax1.set_ylabel('θ [degrees]')
    ax1.set_ylim(0, 90)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

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

def plot_BGSE_fromdata_ontop(spacecraftdata1, spacecraftdata2, whatspacecraft1, whatspacecraft2, whatspacecraft3=None, spacecraftdata3=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(spacecraftdata1[:, 0], label=f'{whatspacecraft1} X', color='red')
    ax1.plot(spacecraftdata2[:, 0], label=f'{whatspacecraft2} X', color='orange')

    ax2.plot(spacecraftdata1[:, 1], label=f'{whatspacecraft1} Y', color='red')
    ax2.plot(spacecraftdata2[:, 1], label=f'{whatspacecraft2} Y', color='orange')

    ax3.plot(spacecraftdata1[:, 2], label=f'{whatspacecraft1} Z', color='red')
    ax3.plot(spacecraftdata2[:, 2], label=f'{whatspacecraft2} Z', color='orange')

    if spacecraftdata3 is not None:
        ax1.plot(spacecraftdata3[:, 0], label=f'{whatspacecraft3} X', color='blue')
        ax2.plot(spacecraftdata3[:, 1], label=f'{whatspacecraft3} Y', color='blue')
        ax3.plot(spacecraftdata3[:, 2], label=f'{whatspacecraft3} Z', color='blue')


    plt.tight_layout()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()
