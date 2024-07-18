import netCDF4 as nc
from icecream import ic
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 10})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from datetime import datetime, timedelta
from utils import format_units, mkticks

datafile = 'C:/Users/sarah.auriemma/Desktop/Data_new/gk2a/pd/gk2a_ksem_pd_e_1m_le1_20240510.nc'
# datafile = 'C:/Users/sarah.auriemma/Desktop/Data_new/gk2a
# /Sarah_KSEM_electron/gk2a_ksem_pd_e_1m_le1_20230226.nc'

num_input_files = 1
CHANNELS_TO_PLOT = 6  # 10 avail?
INCLUDE_CHANNEL_1 = False  # in v1.0.2 (pre feb 2021, channel 1 has a data
# issue)
EXTEND_PLOT_HOURS = 6  # Make room for the legend by extending the plot by
# this many hours

# Colors array for channels 1 to 10
colors = [
    '#A52A2A',  # brown
    '#FF0000',  # red
    '#FFA500',  # orange
    '#F2F200',  # adjusted yellow (darker)
    '#ADFF2F',  # greenyellow
    # '#007300',  # forest green
    '#00b200',  # medium green
    # '#007FC6',  # adjusted blue (lighter)
    '#00c6c1',  # adjusted blue (lighter)
    '#8A2BE2',  # adjusted darkviolet
    '#4B0082',  # adjusted indigo
    '#000000',  # black for E11, but we'll use it for E10 here
]

# Initialize dataset
gk2a_ksem_pd_e_1m_dataset = nc.Dataset(datafile)
ic(gk2a_ksem_pd_e_1m_dataset)
ksem_variables = gk2a_ksem_pd_e_1m_dataset.variables.keys()
ic(ksem_variables)
ic(gk2a_ksem_pd_e_1m_dataset['E2_QEF'][:])
times = gk2a_ksem_pd_e_1m_dataset['Time_Tag']
# ic(times)
epoch = datetime(2000, 1, 1, 12, 0)  # Epoch is January 1, 2000, at 12:00 PM
dt = np.array([epoch + timedelta(seconds=int(t)) for t in times])
date_str = dt[0].strftime("%Y/%m/%d")

start_date, end_date = min(dt), max(dt)

# To extend plot and make room for legend to the right:
xmin = mdates.date2num(start_date)
xmax = mdates.date2num(end_date + timedelta(hours=EXTEND_PLOT_HOURS))
ic(xmin, xmax)

ymin = 1.E-3
ymax = 2.E6

# arrays to store the channel data and range
e_channel_data = []
e_channel_ranges = []
e_channel_name = []

start_channel = 1 if INCLUDE_CHANNEL_1 else 2
end_channel = CHANNELS_TO_PLOT + 1 if INCLUDE_CHANNEL_1 else \
    CHANNELS_TO_PLOT + 2

# Read channel data:
for i in range(start_channel, end_channel):
    channel_key = f'E{i}'
    e_channel_name.append(channel_key)

    e_channel = gk2a_ksem_pd_e_1m_dataset.variables[channel_key]
    e_channel_data.append(e_channel[:])
    e_channel_ranges.append(e_channel.Short_Description)

ic(e_channel_name)

# Get units and format them to look nice for plots:
e_units = gk2a_ksem_pd_e_1m_dataset['E1'].Units
formatted_e_units = format_units(e_units)
print(formatted_e_units)
ic(e_units)

# plotting starts here

fig = plt.figure(1)
numrow = 4
numcol = 1
gs = gridspec.GridSpec(numrow, numcol)
ax1 = plt.subplot2grid((numrow, numcol), (0, 0), colspan=1, rowspan=3)
# fig, ax = plt.subplots()

for channel in range(len(e_channel_data)):
    color_index = channel + (0 if INCLUDE_CHANNEL_1 else 1)
    label_for_legend = f'{e_channel_name[channel]} (' \
                       f'{e_channel_ranges[channel]})'
    # ax1.plot(dt, e_channel_data[channel], label=e_channel_ranges[channel],
    ax1.plot(dt, e_channel_data[channel], label=label_for_legend,
             color=colors[color_index], linewidth=1.5)

ax1.set_yscale('log')

# formatting of the x-axis dates
# locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
# formatter = mdates.ConciseDateFormatter(locator)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)

hours_locator = mdates.HourLocator()  # Locate each hour
hours_formatter = mdates.DateFormatter('%H')  # Format as 'Hour'
ax1.xaxis.set_major_locator(
    mdates.HourLocator(interval=3))  # Major ticks every 3 hours
ax1.xaxis.set_minor_locator(mdates.HourLocator())  # Minor ticks every hour
ax1.xaxis.set_major_formatter(
    mdates.DateFormatter('%H'))  # Display the hour for major ticks

# Create custom tick labels to include the date change at midnight
tick_locations = [mdates.date2num(start_date + timedelta(hours=i)) for i in
                  range(24 + EXTEND_PLOT_HOURS)]
tick_labels = [''] * len(tick_locations)  # Initialize with empty strings

for i, tick in enumerate(tick_locations):
    tick_time = mdates.num2date(tick)
    if tick_time.hour % 3 == 0:
        tick_labels[i] = tick_time.strftime('%H')
    if tick_time.hour == 0:
        # Replace the '00' hour label with '00\nDate' to mark the new day
        tick_labels[i] = tick_time.strftime('00\n%m/%d')

# Apply the custom tick locations and labels
ax1.set_xticks(tick_locations)
ax1.set_xticklabels(tick_labels)

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

# Add legend, labels, and title as per the first plot's style
# ax.legend(title='Energy Ranges', loc='upper right', bbox_to_anchor=(1.15,
# 1), fontsize=10)
ax1.legend(loc='upper right', fontsize=10, frameon=True)
# ax1.set_ylabel(f'Electrons ({formatted_e_units})', fontsize=12)
ax1.set_ylabel(f'electrons/cm$^2$-s-str-keV', fontsize=12)
ax1.set_xlabel('UT [hours]', fontsize=12)
# plt.title(f'GK2A : KSEM 1-minute average Electron Flux - {date_str}',
# fontsize=14)
plt.suptitle(f'GK2A : KSEM 1-minute avg electron flux - {date_str}',
             fontsize=14)

# Adjust the layout and display the plot
plt.tight_layout()
date_str_for_saving = dt[0].strftime("%Y-%m-%d")
plt.savefig(f'GK2A_ksemFlux_{date_str_for_saving}.png', bbox_inches='tight')

plt.show()
