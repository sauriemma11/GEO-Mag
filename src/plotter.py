from coord_transform import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import *
import datetime as dt

"""
NOTE - Color of spacecraft should be same across all plotting functions
GOES17 : red
GOES18 : orange
SOSMAG : blue
"""


def plot_BGSE_fromdata_ontop(goes_time, goes17_spacecraft_data=None,
                             goes18_spacecraft_data=None, whatsc_goes17=None,
                             whatsc_goes18=None, whatsc_gk2a=None,
                             gk2a_spacecraft_data=None, date_str=None,
                             save_path=None, noonmidnighttime_dict=None):
    """
    Plot the B_GSE data from multiple spacecraft on top of each other.
    ** Note: All 3 spacecraft are optional, but at least one must be provided.

    :param goes17_spacecraft_data: Data for the first spacecraft to plot (
    optional).
    :param goes18_spacecraft_data: Data for the second spacecraft to plot (
    optional).
    :param whatsc_goes17: A label for the first spacecraft (optional).
    :param whatsc_goes18: A label for the second spacecraft (optional).
    :param whatsc_gk2a: A label for the third spacecraft (optional).
    :param gk2a_spacecraft_data: Data for the third spacecraft (optional).
    :param date_str: A string representing the date (e.g., "YYYY-MM-DD").
    :param save_path: The file path to save the generated plot (optional).

    Example:
    plot_BGSE_fromdata_ontop(goes17_spacecraft_data, goes18_spacecraft_data,
    whatsc_goes17, whatsc_goes18, whatsc_gk2a, gk2a_spacecraft_data,
    date_str, save_path)
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    if date_str is not None:
        ax1.set_title(f'B Field $GSE$, {date_str}')

    if goes17_spacecraft_data is not None and whatsc_goes17 is not None:
        ax1.plot(goes_time, goes17_spacecraft_data[:, 0],
                 label=f'{whatsc_goes17} ', color='red')
        ax2.plot(goes_time, goes17_spacecraft_data[:, 1],
                 label=f'{whatsc_goes17} ', color='red')
        ax3.plot(goes_time, goes17_spacecraft_data[:, 2],
                 label=f'{whatsc_goes17} ', color='red')

    if goes18_spacecraft_data is not None and whatsc_goes18 is not None:
        ax1.plot(goes_time, goes18_spacecraft_data[:, 0],
                 label=f'{whatsc_goes18} ', color='orange')
        ax2.plot(goes_time, goes18_spacecraft_data[:, 1],
                 label=f'{whatsc_goes18} ', color='orange')
        ax3.plot(goes_time, goes18_spacecraft_data[:, 2],
                 label=f'{whatsc_goes18} ', color='orange')

    if gk2a_spacecraft_data is not None and whatsc_gk2a is not None:
        ax1.plot(goes_time, gk2a_spacecraft_data[:, 0], label=f'{whatsc_gk2a}',
                 color='blue')
        ax2.plot(goes_time, gk2a_spacecraft_data[:, 1], label=f'{whatsc_gk2a}',
                 color='blue')
        ax3.plot(goes_time, gk2a_spacecraft_data[:, 2], label=f'{whatsc_gk2a}',
                 color='blue')

    if noonmidnighttime_dict:
        y_annotation = 10

        # unpack dict
        if 'gk2a' in noonmidnighttime_dict:
            gk2a_noon = noonmidnighttime_dict['gk2a']['noon']
            gk2a_midnight = noonmidnighttime_dict['gk2a']['midnight']
            ax1.annotate('M',
                         xy=(mdates.date2num(gk2a_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='blue', fontsize=12,
                         annotation_clip=True)
            ax1.annotate('N',
                         xy=(mdates.date2num(gk2a_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='blue', fontsize=12,
                         annotation_clip=False)

        if 'g17' in noonmidnighttime_dict:
            g17_noon = noonmidnighttime_dict['g17']['noon']
            g17_midnight = noonmidnighttime_dict['g17']['midnight']
            ax1.annotate('N',
                         xy=(mdates.date2num(g17_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='red',
                         fontsize=12,
                         annotation_clip=False)
            ax1.annotate('M',
                         xy=(mdates.date2num(g17_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='red',
                         fontsize=12,
                         annotation_clip=True)

        if 'g18' in noonmidnighttime_dict:
            g18_noon = noonmidnighttime_dict['g18']['noon']
            g18_midnight = noonmidnighttime_dict['g18']['midnight']
            ax1.annotate('M',
                         xy=(mdates.date2num(g18_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='orange',
                         fontsize=12,
                         annotation_clip=True)
            ax1.annotate('N',
                         xy=(mdates.date2num(g18_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='orange',
                         fontsize=12,
                         annotation_clip=False)

    ax1.legend(loc='upper right')

    ax1.set_ylabel('$B_x$ [nT]')
    ax2.set_ylabel('$B_y$ [nT]')
    ax3.set_ylabel('$B_z$ [nT]')

    ax3.set_xlabel('Time [h]')

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    # Only show 3rd panel x axis labels:
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    plt.tight_layout()
    # TODO: Fix save file, it is not working currently.
    if save_path:
        save_file_as = f'BGSE_{date_str}'
        plt.savefig(f"{save_path}/{save_file_as}.png")
        print(f'fig saved as {save_file_as}')
        plt.show()
    else:
        plt.show()


def plot_magnetic_inclination_over_time_3sc(date_str, goes_time,
                                            goes17_data=None, goes18_data=None,
                                            gk2a_data=None, save_path=None,
                                            noonmidnighttime_dict=None):
    """
    Plot magnetic inclination angle (θ) for multiple spacecraft over time.
    ** Note: All 3 s/c are optional, but at least one must be provided.

    :param date_str: A string representing the date (e.g., "YYYY-MM-DD").
    :param goes_time: The timestamp data for the plotted time.
    :param goes17_data: Data for GOES-17 (optional).
    :param goes18_data: Data for GOES-18 (optional).
    :param gk2a_data: Data for SOSMAG (optional).
    :param save_path: The file path to save the generated plot (optional).
    :param noonmidnighttime_dict: OPTIONAL, data dictionary storing noon and
    mignight times of spacecraft for plotting
    Example:
    plot_magnetic_inclination_over_time_3sc(date_str, goes_time,
    goes17_data, goes18_data, gk2a_data, save_path)
    """
    fig, (ax1) = plt.subplots()

    if goes17_data is not None:
        goes17_theta = calculate_magnetic_inclination_angle_VDH(goes17_data[:, 0],goes17_data[:, 1],goes17_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes17_theta), label='G17', color='red')

    if goes18_data is not None:
        goes18_theta = calculate_magnetic_inclination_angle_VDH(goes18_data[:, 0],goes18_data[:, 1],goes18_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes18_theta), label='G18', color='orange')

    if gk2a_data is not None:
        gk2a_theta = calculate_magnetic_inclination_angle_VDH(gk2a_data[:, 0],
                                                              gk2a_data[:, 1],
                                                              gk2a_data[:, 2])
        ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG',
                 color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ), {date_str}')
    ax1.set_ylabel('θ [degrees]')
    ax1.set_ylim(0, 90)

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    # noon/midnight times plotting:
    if noonmidnighttime_dict:
        y_annotation = 10

        # unpack dict
        if 'gk2a' in noonmidnighttime_dict:
            gk2a_noon = noonmidnighttime_dict['gk2a']['noon']
            gk2a_midnight = noonmidnighttime_dict['gk2a']['midnight']
            ax1.annotate('M',
                         xy=(mdates.date2num(gk2a_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='blue', fontsize=12,
                         annotation_clip=True)
            ax1.annotate('N',
                         xy=(mdates.date2num(gk2a_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='blue', fontsize=12,
                         annotation_clip=False)

        if 'g17' in noonmidnighttime_dict:
            g17_noon = noonmidnighttime_dict['g17']['noon']
            g17_midnight = noonmidnighttime_dict['g17']['midnight']
            ax1.annotate('N',
                         xy=(mdates.date2num(g17_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='red',
                         fontsize=12,
                         annotation_clip=False)
            ax1.annotate('M',
                         xy=(mdates.date2num(g17_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='red',
                         fontsize=12,
                         annotation_clip=True)

        if 'g18' in noonmidnighttime_dict:
            g18_noon = noonmidnighttime_dict['g18']['noon']
            g18_midnight = noonmidnighttime_dict['g18']['midnight']
            ax1.annotate('M',
                         xy=(mdates.date2num(g18_midnight), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='orange',
                         fontsize=12,
                         annotation_clip=True)
            ax1.annotate('N',
                         xy=(mdates.date2num(g18_noon), y_annotation),
                         xytext=(-15, 10),
                         textcoords='offset points', color='orange',
                         fontsize=12,
                         annotation_clip=False)

    ax1.legend(loc='center right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.show()
    else:
        plt.show()


# TODO: make this more general/WORK

def plot_magnetic_field_difference(goes_time, goes_data, gk2a_data, date_str,
                                   use_omni, what_model, what_spacecraft,
                                   show_figs=True, save_figs=False):
    gk2a_time_diff = calculate_time_difference(128.2, 'E')
    g18_time_diff = calculate_time_difference(137.2)

    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    date_obj_previous_day = date_obj - dt.timedelta(
        days=1)  # For plotting noon time GK2A

    midnight_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 0,
                                0)

    noon_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0)
    noon_time_GK2A = dt.datetime(date_obj_previous_day.year,
                                 date_obj_previous_day.month,
                                 date_obj_previous_day.day, 12, 0)

    gk2a_midnight_time = midnight_time + dt.timedelta(hours=gk2a_time_diff)
    g18_midnight_time = midnight_time + dt.timedelta(hours=g18_time_diff)
    gk2a_noon_time = noon_time_GK2A + dt.timedelta(hours=gk2a_time_diff)
    g18_noon_time = noon_time + dt.timedelta(hours=g18_time_diff)

    print(gk2a_midnight_time, gk2a_noon_time, "GK2A")
    print(g18_midnight_time, g18_noon_time, "G18")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    if use_omni:
        title = f'(SOSMAG - {what_model}), ({what_spacecraft} - ' \
                f'{what_model}), using OMNI \n{date_str}'
    else:
        title = f'(SOSMAG - {what_model}), ({what_spacec
        raft} - {what_model}) \n{date_str}'

    ax1.set_title(title)

    y_annotation = 10

    ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12,
                 annotation_clip=True)
    ax1.annotate('M', xy=(mdates.date2num(g18_midnight_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12,
                 annotation_clip=True)
    ax1.annotate('N', xy=(mdates.date2num(g18_noon_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12,
                 annotation_clip=False)
    ax1.annotate('N', xy=(mdates.date2num(gk2a_noon_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12,
                 annotation_clip=False)

    # print(gk2a_midnight_time)
    # print(type(goes_time[2]))
    ax1.plot(goes_time, goes_data[:, 0], 'r')
    ax2.plot(goes_time, goes_data[:, 1], 'r')
    ax3.plot(goes_time, goes_data[:, 2], 'r')

    ax1.plot(goes_time, gk2a_data[:, 0], 'b')
    ax2.plot(goes_time, gk2a_data[:, 1], 'b')
    ax3.plot(goes_time, gk2a_data[:, 2], 'b')

    ax1.legend(['GOES18', 'SOSMAG'])

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.set_ylabel('B Field GSE, cart. [nT]')
    ax3.set_xlabel('Time [h]')

    plt.tight_layout()

    if show_figs:
        plt.show()

    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}_SOSMAG_' \
                   f'{date_str2}_3plts_OMNI.png'
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{w
        hat_model}/{what_spacecraft}/{what_spacecraft}_SOSMAG_{what_model}_{
            date_str2}_3plts.png'

    if save_figs:
        fig.savefig(filename)

    # Plot total mag field differences
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(goes18_time_1min, subtr)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('|B| [nT]')
    ax1.set_title('Total B field difference for ' + date_str)
    ax1.xaxis.set_major_locator(
        mdates.HourLocator(interval=2))  # Show every 2 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.legend().set_visible(False)
    plt.tight_layout()
    if show_figs:
        plt.show()

    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-' \
                   f'{what_model}-{what_spacecraft}-{what_model}_totalB_{
        date_str2}_OMNI.png'
                   }
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/' \
                   f'{what_spacecraft}/sosmag-{w
        hat_model}-{what_spacecraft}-{what_model}_totalB_{date_str2}.png'

    if save_figs:
        fig.savefig(filename)

    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(goes18_time_1min, gk2a_ts04_diff - goes18_ts04_diff)
    date_str = gk2a_time_1min[0].strftime('%Y-%m-%d')

    ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12)
    ax1.annotate(f'M', xy=(mdates.date2num(g18_midnight_time), y_annotation),
                 xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12)

    title = '(SOSMAG - ' + whatModel + ') - (' + whatSpacecraft + ' - ' +
            whatModel + ')\n{}'.format(
        date_str)
    ax1.set_title(title)
    ax1.set(xlabel='Time [h]', ylabel='B Field GSE [nT]')

    ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1.19, 1), loc='upper right')

    ax1.xaxis.set_major_locator(
        mdates.HourLocator(interval=2))  # show every 2 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    print(goes18_time_1min[:])
    fig.patch.set_facecolor('white')

    fig.patch.set_alpha(0.6)
    # ax1.grid(False)

    plt.tight_layout()

    if show_figs:
        plt.show()

    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-' \
                   f'{what_model}-{what_spacecraft}-{what_model}_GSE_{
        date_str2}_OMNI.png'
                   }
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/' \
                   f'{what_spacecraft}/sosmag-{w
        hat_model}-{what_spacecraft}-{what_model}_GSE_{date_str2}.png'

    if save_figs:
        fig.savefig(filename)
