import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def calculate_magnetic_inclination_angle(bx, by, bz):
    return np.arctan2(bx, np.sqrt(by**2 + bz**2))

def plot_magnetic_inclination_over_time(goes_time, goes_data, gk2a_data, date_str):
    # Calculate θ for GOES and GK2A data
    goes_theta = calculate_magnetic_inclination_angle(goes_data[:, 0], goes_data[:, 1], goes_data[:, 2])
    gk2a_theta = calculate_magnetic_inclination_angle(gk2a_data[:, 0], gk2a_data[:, 1], gk2a_data[:, 2])

    # Create plots for θ over time
    fig, (ax1) = plt.subplots()

    # GOES, red
    # SOSMAG, blue

    ax1.plot(goes_time, np.degrees(goes_theta), label='GOES', color='red')
    ax1.plot(goes_time, np.degrees(gk2a_theta), label='SOSMAG', color='blue')

    ax1.set_title(f'Magnetic Inclination Angle (θ) - {date_str}')
    ax1.set_ylabel('θ [degrees]')


    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.legend()

    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    # Specify the paths to your pickle files
    goes_pickle_path = 'Z:/Data/GOES18/model_outs/20221217/modout_20221217.pickle'
    gk2a_pickle_path = 'Z:/Data/GK2A/model_outputs/20221217/sosmag_modout_2022-12-17.pickle'

    # Load GOES and GK2A data
    goes_data = load_pickle_file(goes_pickle_path)['sat']
    gk2a_data = load_pickle_file(gk2a_pickle_path)['sat_gse']

    # Load the corresponding time data (adjust this based on your data structure)
    goes_time = load_pickle_file(goes_pickle_path)['time_min']
    gk2a_time = load_pickle_file(gk2a_pickle_path)['time_min']

    # You can also specify a date string (e.g., '2022-12-17') for the plot title
    date_str = '2022-12-17'

    # Plot magnetic inclination angle over time for GOES and GK2A
    plot_magnetic_inclination_over_time(goes_time, goes_data, gk2a_data, date_str)