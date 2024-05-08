import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import numpy as np

# Load MPSH plot data
with open('C:/Users/sarah.auriemma/Desktop/Data_new/mpsh_plot_data.pickle',
          'rb') as file:
    mpsh_plot_data = pickle.load(file)

# Load Magnetic Inclination data
with open(
        'C:/Users/sarah.auriemma/Desktop/Data_new/mag_incl_VDH/VDH_2019-05'
        '-14.pickle',
        'rb') as file:
    mag_incl_data = pickle.load(file)

# Assuming the timestamp data is in the format suitable for plotting
# If necessary, convert your timestamps to matplotlib date format
timestamps = mpsh_plot_data['timestamps']
timestamps = mdates.date2num(
    timestamps)  # Convert to matplotlib's date format if not already done

fig, ax1 = plt.subplots()

# Plot MPSH data
ax1.plot_date(timestamps, mpsh_plot_data['AvgIntElectronFlux'][:, 2], '-',
              label='MPSH AvgIntElectronFlux')  # Adjust the index as necessary

# Optionally create a second y-axis for the magnetic inclination data
ax2 = ax1.twinx()

color_map = {'G16': 'red', 'G17': 'orange', 'GK2A': 'blue'}
for satellite, data in mag_incl_data.items():
    if satellite in color_map:
        ax2.plot_date(timestamps, np.degrees(data), '-', label=satellite,
                      color=color_map[satellite])

# Customize plot
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax1.set_xlabel('Time')
ax1.set_ylabel('MPSH Data')
ax2.set_ylabel('Magnetic Inclination Angle (θ)')
fig.autofmt_xdate()  # Format date labels to fit nicely
plt.legend()
plt.show()
