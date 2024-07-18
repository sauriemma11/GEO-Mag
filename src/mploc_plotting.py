import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import to_datetime


def plot_mpause_plots(goes_results, flag_Arr, sw_data, shue_r0, sw_dyn_p, sat_name):
    fig, axs = plt.subplots(6, 1, sharex=True)

    # Magnetic field Hp
    axs[0].plot(goes_results['datetime_values'], goes_results['b_epn'][:, 1], label='Hp [nT]')
    axs[0].set_ylabel('Hp [nT]')
    axs[0].legend()

    # Electron and Ion Ratios with labels on the left and right
    ax1 = axs[1]
    ax2 = ax1.twinx()  # Create a twin Axes sharing the xaxis
    ax1.semilogy(goes_results['datetime_values'], goes_results['electron_ratios'], 'b-', label='Electrons')
    ax2.semilogy(goes_results['datetime_values'], goes_results['ion_ratios'], 'r-', label='Ions')
    ax1.set_ylabel('Electron Ratio')
    ax2.set_ylabel('Ion Ratio')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Shue r0
    axs[2].plot(goes_results['datetime_values'], shue_r0, 'g-', label='Shue r0')
    axs[2].axhline(y=6.6, color='r', linestyle='--', label='6.6 Re')
    axs[2].set_ylabel('Shue r0')
    axs[2].legend()

    # FLAG PLOTTING
    datetime_values = np.array(goes_results['datetime_values'])
    g16_flags = {key: np.array(val) for key, val in flag_Arr.items()}

    # Add a subplot for flags
    axflag = axs[3]
    flag_levels = {'flag_r0': 4, 'flag_b_field': 3, 'flag_electrons': 2, 'flag_ions': 1}
    for flag_name, level in flag_levels.items():
        flag_data = g16_flags[flag_name]
        # Ensure flag_data is a boolean array for indexing
        flag_indices = np.where(flag_data == 1)[0]  # Get indices where flags are true
        axflag.plot(datetime_values[flag_indices], np.full(flag_indices.size, level), '|', markersize=10,
                    label=flag_name)

    axflag.set_yticks(list(flag_levels.values()))
    axflag.set_yticklabels([name.replace('flag_', '').title() for name in flag_levels.keys()])
    axflag.set_ylim(0.5, len(flag_levels) + 0.5)
    axflag.legend()

    # B field components
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 0], 'r-', label='B_GSM X')
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 1], 'g-', label='B_GSM Y')
    axs[4].plot(goes_results['datetime_values'], goes_results['b_gsm'][:, 2], 'b-', label='B_GSM Z')
    axs[4].set_ylabel('B_GSM [nT]')
    axs[4].legend(loc='upper left')

    # Solar Wind properties combined in one plot with two y-axes
    sw_dates = mdates.date2num(to_datetime(sw_data['Epoch']))

    ax4 = axs[5]
    ax5 = ax4.twinx()  # Create a twin Axes sharing the xaxis
    ax4.plot(sw_dates, sw_dyn_p, 'm-', label='SW Density')
    ax5.plot(sw_dates, sw_data['flow_speed'], 'purple', label='SW Speed')
    ax4.set_ylabel('SW Density [nPa]')
    ax5.set_ylabel('SW Speed [km/s]')
    ax4.legend(loc='upper left')
    ax5.legend(loc='upper right')

    plt.gcf().autofmt_xdate()  # Automatically formats the x-dates to look better
    date_format = mdates.DateFormatter('%H')
    ax4.xaxis.set_major_formatter(date_format)

    # Set common labels
    for ax in axs[:-1]:
        ax.label_outer()  # Hide x labels and tick labels for top plots and right y-axis

    axs[-1].set_xlabel('Time [hours]')  # Only set x-label on the last subplot
    plt.tight_layout()
    fig.suptitle(f'Magnetopause Location, {sat_name}')

    plt.show()
