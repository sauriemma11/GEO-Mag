import netCDF4 as nc
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.colors as cm
from datetime import datetime, timedelta
from utils import format_units

datafile = 'C:/Users/sarah.auriemma/Desktop/Data_new/gk2a/pd' \
           '/gk2a_ksem_pd_e_1m_le1_20231002.nc'
gk2a_ksem_pd_e_1m_dataset = nc.Dataset(datafile)
# ic(gk2a_ksem_pd_e_1m_dataset)
ksem_variables = gk2a_ksem_pd_e_1m_dataset.variables.keys()
# ic(ksem_variables)

times = gk2a_ksem_pd_e_1m_dataset['Time_Tag']
# ic(times)
epoch = datetime(2000, 1, 1, 12, 0)  # Epoch is January 1, 2000, at 12:00 PM
dt = np.array([epoch + timedelta(seconds=int(t)) for t in times])
ic(dt[0], dt[-1])
date_str = dt[0].strftime("%Y-%m-%d")

# arrays to store the channel data and ranges
e_channel_data = []
e_channel_ranges = []

for i in range(1, 11):
    channel_key = f'E{i}'
    e_channel = gk2a_ksem_pd_e_1m_dataset.variables[channel_key]
    e_channel_data.append(e_channel[:])
    e_channel_ranges.append(e_channel.Short_Description)

# ic(e_channel_data)
# ic(e_channel_ranges)

e_units = gk2a_ksem_pd_e_1m_dataset['E1'].Units
formatted_e_units = format_units(e_units)
print(formatted_e_units)
ic(e_units)

fig, ax = plt.subplots()

for channel in range(10):
    ax.plot(dt, e_channel_data[channel], label=f'{e_channel_ranges[channel]}')

ax.set_yscale('log')

# Improve the formatting of the x-axis dates
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Add legend, labels, and title
ax.legend(title='Energy Ranges', loc='upper right', bbox_to_anchor=(1.15, 1))
ax.set_ylabel(f'Electrons ({formatted_e_units})')
ax.set_xlabel('Time')
plt.suptitle('GK2A : KSEM 1-minute average Electron Flux',
             fontsize=14)  # Main title
ax.set_title(f'{date_str}', fontsize=10)  # Subtitle with date

# # Add grid for better readability
# ax.grid(True, which="both", ls="--", linewidth=0.5)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()
