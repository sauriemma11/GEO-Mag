import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import to_datetime


def make_mpause_plots(goes_results, flag_Arr, sw_data, shue_r0, sw_dyn_p, sat_name, sw_data_via):
    min_shue_r0 = np.nanmin(shue_r0)

    fig, axs = plt.subplots(6, 1, sharex=True)

    # Font sizes
    axis_font_size = 9
    legend_font_size = 6

    # Magnetic field Hp
    axs[0].plot(goes_results['datetime_values'], goes_results['b_epn'][:, 1], label='Hp', color='k')
    axs[0].set_ylabel('Hp\n[nT]', fontsize=axis_font_size)
    axs[0].legend(fontsize=legend_font_size)

    # Electron and Ion Ratios with labels on the left and right
    ax1 = axs[1]
    ax2 = ax1.twinx()  # Create a twin Axes sharing the xaxis
    ax1.semilogy(goes_results['datetime_values'], goes_results['electron_ratios'], 'b-', label='Electrons')
    ax2.semilogy(goes_results['datetime_values'], goes_results['ion_ratios'], 'r-', label='Ions')
    ax1.set_ylabel('Electron Ratio', fontsize=axis_font_size)
    ax2.set_ylabel('Ion Ratio', fontsize=axis_font_size)
    ax1.legend(loc='upper left', fontsize=legend_font_size)
    ax2.legend(loc='upper right', fontsize=legend_font_size)

    # Shue r0
    axs[2].plot(goes_results['datetime_values'], shue_r0, 'g-', label='Shue r0', color='green')
    axs[2].axhline(y=6.6, color='r', linestyle='--', label='6.6 Re')
    axs[2].set_ylabel('Shue r0', fontsize=axis_font_size)
    axs[2].legend(fontsize=legend_font_size)

    # FLAG PLOTTING
    colors = {
        'r0': 'green',
        'b_field': 'black',
        'electrons': 'blue',
        'ions': 'red'
    }

    datetime_values = np.array(goes_results['datetime_values'])
    g16_flags = {key: np.array(val) for key, val in flag_Arr.items()}

    # Add a subplot for flags
    axflag = axs[3]
    flag_levels = {'flag_r0': 4, 'flag_b_field': 3, 'flag_electrons': 2, 'flag_ions': 1}
    for flag_name, level in flag_levels.items():
        flag_data = g16_flags[flag_name]
        # Ensure flag_data is a boolean array for indexing
        flag_indices = np.where(flag_data == 1)[0]  # Get indices where flags are true
        modified_name = flag_name.replace('flag_', '')
        color = colors.get(modified_name, 'black')  # Default to black if not found
        axflag.plot(datetime_values[flag_indices], np.full(flag_indices.size, level), '|', markersize=3,
                    label=modified_name, color=color)

    axflag.set_yticks(list(flag_levels.values()))
    axflag.set_yticklabels([name.replace('flag_', '').title() for name in flag_levels.keys()], fontsize=axis_font_size)
    axflag.set_ylim(0.5, len(flag_levels) + 0.5)

    axflag.yaxis.tick_right()
    axflag.yaxis.set_label_position("left")
    axflag.set_ylabel("Flags", fontsize=axis_font_size)

    # B field components
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 0], 'r-', label='X')
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 1], 'g-', label='Y')
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 2], 'b-', label='Z')
    axs[4].set_ylabel('B_GSM\n[nT]', fontsize=axis_font_size)
    axs[4].legend(loc='upper left', fontsize=legend_font_size)

    # Solar Wind properties combined in one plot with two y-axes
    sw_dates = mdates.date2num(to_datetime(sw_data['Epoch']))

    ax4 = axs[5]
    ax5 = ax4.twinx()  # Create a twin Axes sharing the xaxis
    ax4.plot(sw_dates, sw_dyn_p, 'hotpink', label='SW Density')
    ax5.plot(sw_dates, sw_data['flow_speed'], 'blueviolet', label='SW Speed')
    ax4.set_ylabel('SW Density\n[nPa]', fontsize=axis_font_size)
    ax5.set_ylabel('SW Speed\n[km/s]', fontsize=axis_font_size)
    ax4.legend(loc='upper left', fontsize=legend_font_size)
    ax5.legend(loc='upper right', fontsize=legend_font_size)

    plt.gcf().autofmt_xdate()  # Automatically formats the x-dates to look better
    date_format = mdates.DateFormatter('%H')
    ax4.xaxis.set_major_formatter(date_format)

    # Set common labels
    for ax in axs[:-1]:
        ax.label_outer()  # Hide x labels and tick labels for top plots and right y-axis

    axs[-1].set_xlabel('Time [hours]')  # Only set x-label on the last subplot

    datetime_values = np.array(goes_results['datetime_values'])
    datetime_strings = [d.strftime('%Y-%m-%d %H:%M') for d in datetime_values]

    start_date = datetime_values[0]
    end_date = datetime_values[-1]
    date_format = '%Y-%m-%d'
    # Determine if the date range is for a single day or multiple days
    if start_date.date() == end_date.date():
        date_label = start_date.strftime(date_format)
    else:
        date_label = f'{start_date.strftime(date_format)} to {end_date.strftime(date_format)}'

    # plt.tight_layout()
    # fig.suptitle(f'{sat_name}, {date_label}'
    #              f'\n{sw_data_via}')

    # Set the main title with normal size and the subtitle with smaller font
    plt.figtext(0.5, 0.97, f'{sat_name}, {date_label}', ha='center', va='center', fontsize=axis_font_size + 1)
    plt.figtext(0.5, 0.94, f'*Shue et al. 1998 using {sw_data_via}', ha='center', va='center', fontsize=axis_font_size)
    plt.figtext(0.5, 0.91, f'Min R0 for time range: {round(min_shue_r0, 1)}', ha='center', va='center',
                fontsize=axis_font_size - 1)

    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.125, right=0.9, hspace=0.185, wspace=0.2)

    plt.show()
