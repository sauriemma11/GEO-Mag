import utils
from coord_transform import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from utils import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from datetime import datetime as dtm

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
        goes17_theta = calculate_magnetic_inclination_angle_VDH(
            goes17_data[:, 0], goes17_data[:, 1], goes17_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes17_theta), label='G17', color='red')

    if goes18_data is not None:
        goes18_theta = calculate_magnetic_inclination_angle_VDH(
            goes18_data[:, 0], goes18_data[:, 1], goes18_data[:, 2])
        ax1.plot(goes_time, np.degrees(goes18_theta), label='G18',
                 color='orange')

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

# def plot_magnetic_field_difference(goes_time, goes_data, gk2a_data,
# date_str, use_omni, what_model,what_spacecraft, show_figs=True,
# save_figs=False):
#     gk2a_time_diff = calculate_time_difference(128.2, 'E')
#     g18_time_diff = calculate_time_difference(137.2)
#
#     date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
#     date_obj_previous_day = date_obj - dt.timedelta(
#         days=1)  # For plotting noon time GK2A
#
#     midnight_time = dt.datetime(date_obj.year, date_obj.month,
#     date_obj.day, 0,
#                                 0)
#
#     noon_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day,
#     12, 0)
#     noon_time_GK2A = dt.datetime(date_obj_previous_day.year,
#                                  date_obj_previous_day.month,
#                                  date_obj_previous_day.day, 12, 0)
#
#     gk2a_midnight_time = midnight_time + dt.timedelta(hours=gk2a_time_diff)
#     g18_midnight_time = midnight_time + dt.timedelta(hours=g18_time_diff)
#     gk2a_noon_time = noon_time_GK2A + dt.timedelta(hours=gk2a_time_diff)
#     g18_noon_time = noon_time + dt.timedelta(hours=g18_time_diff)
#
#     print(gk2a_midnight_time, gk2a_noon_time, "GK2A")
#     print(g18_midnight_time, g18_noon_time, "G18")
#
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
#
#     if use_omni:
#         title = f'(SOSMAG - {what_model}), ({what_spacecraft} - ' \
#                 f'{what_model}), using OMNI \n{date_str}'
#     else:
#         title = f'(SOSMAG - {what_model}), ({what_spacecraft} - {
#         what_model}) \n{date_str}'
#
#     ax1.set_title(title)
#
#     y_annotation = 10
#
#     ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='blue', fontsize=12,
#                  annotation_clip=True)
#     ax1.annotate('M', xy=(mdates.date2num(g18_midnight_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='red', fontsize=12,
#                  annotation_clip=True)
#     ax1.annotate('N', xy=(mdates.date2num(g18_noon_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='red', fontsize=12,
#                  annotation_clip=False)
#     ax1.annotate('N', xy=(mdates.date2num(gk2a_noon_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='blue', fontsize=12,
#                  annotation_clip=False)
#
#     # print(gk2a_midnight_time)
#     # print(type(goes_time[2]))
#     ax1.plot(goes_time, goes_data[:, 0], 'r')
#     ax2.plot(goes_time, goes_data[:, 1], 'r')
#     ax3.plot(goes_time, goes_data[:, 2], 'r')
#
#     ax1.plot(goes_time, gk2a_data[:, 0], 'b')
#     ax2.plot(goes_time, gk2a_data[:, 1], 'b')
#     ax3.plot(goes_time, gk2a_data[:, 2], 'b')
#
#     ax1.legend(['GOES18', 'SOSMAG'])
#
#     ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#
#     ax1.set_ylabel('B Field GSE, cart. [nT]')
#     ax3.set_xlabel('Time [h]')
#
#     plt.tight_layout()
#
#     if show_figs:
#         plt.show()
#
#     if use_omni:
#         filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}_SOSMAG_{
#         date_str2}_3plts_OMNI.png'
#     else:
#         filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/{
#         what_spacecraft}/{what_spacecraft}_SOSMAG_{what_model}_{
#         date_str2}_3plts.png'
#
#     if save_figs:
#         fig.savefig(filename)
#
#     # Plot total mag field differences
#     fig, (ax1) = plt.subplots(1, 1)
#     ax1.plot(goes18_time_1min, subtr)
#     ax1.set_xlabel('Time')
#     ax1.set_ylabel('|B| [nT]')
#     ax1.set_title('Total B field difference for ' + date_str)
#     ax1.xaxis.set_major_locator(
#         mdates.HourLocator(interval=2))  # Show every 2 hours
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     ax1.legend().set_visible(False)
#     plt.tight_layout()
#     if show_figs:
#         plt.show()
#
#     if use_omni:
#         filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-' \
#                    f'{what_model}-{what_spacecraft}-{what_model}_totalB_{
#         date_str2}_OMNI.png'
#                    }
#     else:
#         filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/' \
#                    f'{what_spacecraft}/sosmag-{w
#         hat_model}-{what_spacecraft}-{what_model}_totalB_{date_str2}.png'
#
#     if save_figs:
#         fig.savefig(filename)
#
#     fig, (ax1) = plt.subplots(1, 1)
#     ax1.plot(goes18_time_1min, gk2a_ts04_diff - goes18_ts04_diff)
#     date_str = gk2a_time_1min[0].strftime('%Y-%m-%d')
#
#     ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='blue', fontsize=12)
#     ax1.annotate(f'M', xy=(mdates.date2num(g18_midnight_time), y_annotation),
#                  xytext=(-15, 10),
#                  textcoords='offset points', color='red', fontsize=12)
#
#     title = '(SOSMAG - ' + whatModel + ') - (' + whatSpacecraft + ' - ' +
#             whatModel + ')\n{}'.format(
#         date_str)
#     ax1.set_title(title)
#     ax1.set(xlabel='Time [h]', ylabel='B Field GSE [nT]')
#
#     ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1.19, 1), loc='upper right')
#
#     ax1.xaxis.set_major_locator(
#         mdates.HourLocator(interval=2))  # show every 2 hours
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#     print(goes18_time_1min[:])
#     fig.patch.set_facecolor('white')
#
#     fig.patch.set_alpha(0.6)
#     # ax1.grid(False)
#
#     plt.tight_layout()
#
#     if show_figs:
#         plt.show()
#
#     if use_omni:
#         filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-' \
#                    f'{what_model}-{what_spacecraft}-{what_model}_GSE_{
#         date_str2}_OMNI.png'
#                    }
#     else:
#         filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/' \
#                    f'{what_spacecraft}/sosmag-{w
#         hat_model}-{what_spacecraft}-{what_model}_GSE_{date_str2}.png'
#
#     if save_figs:
#         fig.savefig(filename)


def plot_sc_vs_sc_scatter(x, y, x_label='X-axis', y_label='Y-axis',
                          title='Scatter Plot', lineofbestfit=False):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if lineofbestfit:
        polynomial = utils.calc_line_of_best_fit(x, y)
        x_fit = np.linspace(min(x), max(x), len(x))
        y_fit = polynomial(x_fit)
        plt.plot(x_fit, y_fit, color='red', linewidth=1)

    plt.show()


def plot_components_vs_t89_with_color(spacecraft_name, data_list,
                                      t89_data_list, timestamps,
                                      output_file=None):
    # Unpack x, y, and z components from the data and T89 data
    x_component, y_component, z_component = unpack_components(data_list)
    t89_x_component, t89_y_component, t89_z_component = unpack_components(
        t89_data_list)

    # Debugging prints:
    # print(max(t89_x_component), max(t89_z_component))
    # print(np.argmax(t89_x_component), np.argmax(t89_z_component))
    # print(timestamps[np.argmax(t89_x_component)])

    # Create subplots for the x, y, and z components vs. T89 components
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Convert timestamps to numeric values for coloring
    numeric_timestamps = mdates.date2num(timestamps)

    # Create a Normalize object to map numeric colors to the range [0, 1]
    color_norm = Normalize(vmin=min(numeric_timestamps),
                           vmax=max(numeric_timestamps))

    # Scatter plots for components vs. T89 components with date-based color
    # mapping
    for ax, component, t89_component, label in zip(axs,
                                                   [x_component, y_component,
                                                    z_component],
                                                   [t89_x_component,
                                                    t89_y_component,
                                                    t89_z_component],
                                                   ['X', 'Y', 'Z']):
        scatter = ax.scatter(component, t89_component, c=numeric_timestamps,
                             cmap='viridis', norm=color_norm)
        ax.set_xlabel(f'{spacecraft_name} {label} Component')
        ax.set_ylabel(f'T89 {label} Component')
        ax.set_title(
            f'{spacecraft_name} {label} Component vs T89 {label} Component')
        fig.colorbar(scatter, ax=ax, format=DateFormatter('%m-%d'),
                     label='Date')

    # Show the plot
    plt.tight_layout()

    # Save the plot to the output file if provided
    if output_file:
        plt.savefig(output_file)

    plt.show()


def plot_4_scatter_plots_with_color(g17_mag_data, g17_sub_data, g17_time_list,
                                    gk2a_mag_data, gk2a_sub_data,
                                    gk2a_time_list, output_file=None,
                                    best_fit=False, is_model_subtr=False):
    fig, axs = plt.subplots(2, 2,
                            figsize=(12, 12))  # Creates a 2x2 grid of subplots

    # Convert timestamps to numeric values
    g17_time_numeric = mdates.date2num(g17_time_list)
    gk2a_time_numeric = mdates.date2num(gk2a_time_list)

    # Create a Normalize object to map numeric timestamps to colors
    g17_time_norm = Normalize(vmin=min(g17_time_numeric),
                              vmax=max(g17_time_numeric))
    gk2a_time_norm = Normalize(vmin=min(gk2a_time_numeric),
                               vmax=max(gk2a_time_numeric))

    # colormap
    cmap = 'viridis'

    # Create a ScalarMappable to generate a colorbar
    g17_time_cmap = ScalarMappable(norm=g17_time_norm, cmap=cmap)
    gk2a_time_cmap = ScalarMappable(norm=gk2a_time_norm, cmap=cmap)

    # Scatter plot 1: subtr vs subtr with date-based color mapping for G17
    x, y = g17_sub_data, gk2a_sub_data
    axs[0, 0].scatter(x, y, c=g17_time_numeric,
                      cmap='viridis', norm=g17_time_norm)
    if is_model_subtr:
        axs[0, 0].set_xlabel('G17 |B| (GSE) with T89 model removed')
        axs[0, 0].set_ylabel('GK2A |B| (GSE) with T89 model removed')
    else:
        axs[0, 0].set_xlabel('G17 |B| (GSE) T89 model')
        axs[0, 0].set_ylabel('GK2A |B| (GSE) T89 model')
    fig.colorbar(g17_time_cmap, ax=axs[0, 0], format=DateFormatter(
        '%m'), label='Date')
    if best_fit:
        polynomial = calc_line_of_best_fit(x, y)
        axs[0, 0].plot(x, polynomial(x), color='red')

    # Scatter plot 2: mag vs mag with date-based color mapping for GK2A
    x, y = g17_mag_data, gk2a_mag_data
    axs[0, 1].scatter(x, y, c=gk2a_time_numeric,
                      cmap='viridis', norm=gk2a_time_norm)
    axs[0, 1].set_xlabel('G17 |B| (GSE) observed')
    axs[0, 1].set_ylabel('GK2A |B| (GSE) observed')
    # axs[0, 1].set_title('G17 vs GK2A |B|')
    fig.colorbar(gk2a_time_cmap, ax=axs[0, 1], format=DateFormatter('%m'),
                 label='Date')  # Add colorbar for dates
    if best_fit:
        polynomial = calc_line_of_best_fit(x, y)
        axs[0, 1].plot(x, polynomial(x), color='red')

    # Scatter plot 3: G17; mag vs subr with date-based color mapping for G17
    x, y = g17_sub_data, g17_mag_data
    axs[1, 0].scatter(x, y, c=g17_time_numeric,
                      cmap='viridis', norm=g17_time_norm)
    if is_model_subtr:
        axs[1, 0].set_xlabel('G17 |B| (GSE) with T89 model removed')
    else:
        axs[1, 0].set_xlabel('G17 |B| (GSE) T89 model')
    axs[1, 0].set_ylabel('G17 |B| (GSE) observed')
    # axs[1, 0].set_title('G17; mag vs subr')
    fig.colorbar(g17_time_cmap, ax=axs[1, 0], format=DateFormatter('%m'),
                 label='Date')  # Add colorbar for dates
    if best_fit:
        polynomial = calc_line_of_best_fit(x, y)
        axs[1, 0].plot(x, polynomial(x), color='red')

    # Scatter plot 4: GK2A; mag vs subr with date-based color mapping for GK2A
    x, y = gk2a_sub_data, gk2a_mag_data
    axs[1, 1].scatter(x, y, c=gk2a_time_numeric,
                      cmap='viridis', norm=gk2a_time_norm)
    if is_model_subtr:
        axs[1, 1].set_xlabel('GK2A |B| with T89 model removed')
    else:
        axs[1, 1].set_xlabel('GK2A |B| (GSE) T89 model')
    axs[1, 1].set_ylabel('GK2A |B| (GSE) observed')
    # axs[1, 1].set_title('GK2A; mag vs subr')
    fig.colorbar(gk2a_time_cmap, ax=axs[1, 1], format=DateFormatter('%m'),
                 label='Date')  # Add colorbar for dates
    if best_fit:
        polynomial = calc_line_of_best_fit(x, y)
        axs[1, 1].plot(x, polynomial(x), color='red')

    # TODO: Fix axes limits
    # axs[1,0].set_xlim(75,130)
    # axs[1,0].set_ylim(25,175)
    # axs[1,1].set_xlim(75,130)
    # axs[1,1].set_ylim(25,175)

    # Remove top and right borders from all panels, apply x y axes limits
    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save the plot to the output file if provided
    if output_file:
        plt.savefig(output_file)

    # Show the plot (optional)
    plt.show()
