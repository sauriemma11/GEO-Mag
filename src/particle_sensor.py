import netCDF4 as nc
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.colors as cm
from datetime import datetime, timedelta

# Channel labels and y-axis limits
DIFF_ELECTRON_CHANNEL_LABELS = ["E1S", "E2", "E3", "E4", "E5", "E6", "E7",
                                "E8", "E9", "E10"]
DIFF_ELECTRON_YAXIS_LIMITS = [10 ** -1, 10 ** 7]

tel = 4  # Change this to what telescope you want to look at
tel_NUM_indata = tel - 1

rainbow_colors = [
    'xkcd:black', 'xkcd:indigo', 'xkcd:blue', 'xkcd:cyan',
    'xkcd:mint green', 'xkcd:lemon lime', 'xkcd:yellow',
    'xkcd:tangerine', 'xkcd:red'
]

datafile = 'C:/Users/sarah.auriemma/Desktop/sci_mpsh-l2' \
           '-avg1m_g18_d20220804_v2-0-0.nc'
mpsh_Dataset = nc.Dataset(datafile)
mpsh_variables = mpsh_Dataset.variables.keys()
ic(mpsh_variables)

times = mpsh_Dataset['time'][:]
epoch = datetime(2000, 1, 1, 12, 0)  # Epoch is January 1, 2000, at 12:00 PM
dt = np.array([epoch + timedelta(seconds=int(t)) for t in times])
ic(dt[0], dt[-1])
date_str = dt[0].strftime("%Y-%m-%d")

# effective_energies_etel4 = mpsh_Dataset['DiffElectronEffectiveEnergy']
effective_energies_etel = mpsh_Dataset['DiffElectronEffectiveEnergy'][
                          tel_NUM_indata, :]
energies = effective_energies_etel[:].data
# For labelling the energies:
rounded_energies = np.around(energies).astype(int)
label_energy_levels = [f"{level}" for level in rounded_energies]

# Electron Flux
fluxes_electron = mpsh_Dataset['AvgDiffElectronFlux']
ic(fluxes_electron)
electron_flux_single_telescope = fluxes_electron[:, tel_NUM_indata, :]

# ic(electron_flux_single_telescope.shape) ic|
# electron_flux_single_telescope.shape: (1440, 10)


# Set up the figure and axis for plotting
fig, ax = plt.subplots()

# for channel in range(electron_flux_single_telescope.shape[1]):
for channel in range(3):
    ax.plot(dt, electron_flux_single_telescope[:, channel],
            label=f'{label_energy_levels[channel]} keV')

# for i, color in zip(range(electron_flux_single_telescope.shape[1]),
# rainbow_colors):
#     ax.plot(dt, electron_flux_single_telescope[:, i],
#     label=DIFF_ELECTRON_CHANNEL_LABELS[i], color=color)

ax.set_yscale('log')
# ax.set_ylim(DIFF_ELECTRON_YAXIS_LIMITS)

# Improve the formatting of the x-axis dates
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Add legend, labels, and title
ax.legend(title='Channels', loc='upper right', bbox_to_anchor=(1.15, 1))
ax.set_ylabel(r'Electrons (cm$^{-2}$ sr$^{-1}$ s$^{-1}$ keV$^{-1}$)')
ax.set_xlabel('Time')
ax.set_title(f'SEISS MPS-HI Fluxes - Electron Telescope {tel}')

# # Add grid for better readability
# ax.grid(True, which="both", ls="--", linewidth=0.5)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# # Create subplots
# fig, ax_electron = plt.subplots(figsize=(10, 6), dpi=200)
#
# # Set up the locator and formatter for the x-axis
# locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# formatter = mdates.ConciseDateFormatter(locator)
#
# # Plot electron fluxes
# for zone_index in range(fluxes_electron.shape[1]):  # Assuming zone is the
# second dimension
#     fluxes_electron_zone = fluxes_electron[:, zone_index, :]
#     for energy_channel_index in range(fluxes_electron_zone.shape[1]):
#         ax_electron.plot(dt, fluxes_electron_zone[:, energy_channel_index],
#                          label=f'Zone {zone_index+1}, Energy Channel {
#                          energy_channel_index+1}')
#
# # Set x-axis label
# ax_electron.set_xlabel('Time')
# # Set y-axis label
# ax_electron.set_ylabel('Electron Flux (electrons/(cm^2 sr keV s))')
# # Set plot title
# ax_electron.set_title('Time-averaged Electron Fluxes')
#
# # Set up the locator and formatter for the x-axis
# ax_electron.xaxis.set_major_locator(locator)
# ax_electron.xaxis.set_major_formatter(formatter)
# plt.xticks(rotation=45)
#
# # Add legend
# ax_electron.legend()
#
# # Show plot
# plt.tight_layout()
# plt.show()
# '''
# seiss plot
# '''
# var1 = 'AvgIntElectronFlux'
# var3 = 'L2_SciData_TimeStamp'
# var2 = 'time'
# old = files[0]
# ufiles = []
# fluxes = []
# times = []
#
# # d = Dataset('/Users/aspen.davis/Documents/radiation_belt/G17/sci_mpsh-l2
# -avg5m_g17_d20220331_v1-0-3.nc')
# # time1_temp=d.variables[var3][:]
# # times.append(time1_temp)
# # flux1_temp=d.variables[var1][:]
# # fluxes.append(flux1_temp)
# for i in files:
#     # print(i)
#     # if i.split('')[-2] != old.split('')[-2]:
#     d = Dataset(i)
#     flux1 = d.variables[var1][:]
#     fluxes.append(flux1)
#     time1 = d.variables[var2][:]
#     times.append(time1)
#     ufiles.append(i)
#     old = i
#
# nfluxes = np.asarray(fluxes)
# print(np.shape(fluxes))
# q = np.where(nfluxes < -1e30)
# nfluxes[q] = np.nan
# print(np.shape(nfluxes))
# afluxes = np.reshape(nfluxes, (
# 28816, 5))  # multiply by how many days you are doing :) (28816,5)
# times = np.asarray(times)
# times = np.reshape(times, (288 * 16))  # same
# atimes = j2000_sec_to_datetime(times)
# # use channel 4, which is index 3
# # %matplotlib notebook
# from matplotlib.offsetbox import AnchoredText
#
# fig, (ax0, ax1) = plt.subplots(2, figsize=(12, 6))
# fig.tight_layout(pad=-0.5)
# ax0.plot(atimes, afluxes[:, 3])
# ax0.set_yscale('log')
# ax0.set_ylim([10, 10000])
# date_form = DateFormatter('%m-%d')
# ax0.xaxis.set_major_formatter(date_form)
# at = AnchoredText(
#     "GOES-18", prop=dict(size=20), frameon=True, loc='upper right')
# at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# ax0.add_artist(at)
# ax0.tick_params(axis='y', labelsize=20)
# ax0.xaxis.set_ticklabels([])
# ax0.margins(x=0)
# ax1.margins(x=0)
# ax0.axhline(y=1000, color='r', linestyle='--')
# ax1.plot(atimes16, afluxes16[:, 3])
# ax1.set_yscale('log')
# ax1.set_ylim([10, 10000])
# ax1.xaxis.set_major_formatter(date_form)
# ax1.set_xlabel('Date [UT]', fontsize=20)
# at2 = AnchoredText(
#     "GOES-16", prop=dict(size=20), frameon=True, loc='upper right')
# at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# ax1.add_artist(at2)
# ax1.tick_params(axis='both', labelsize=20
#                 )
# ax1.axhline(y=1000, color='r', linestyle='--')
# ax1.set_ylabel(
#     'Particle Flux \n [particles $\cdot cm^{-2} \cdot s^{-1} \cdot sr^{
#     -1}$]',
#     fontsize=14)
# ax0.set_ylabel(
#     'Particle Flux \n [particles $\cdot cm^{-2} \cdot s^{-1} \cdot sr^{
#     -1}$]',
#     fontsize=14)
