import matplotlib.pyplot as plt
import utils
import data_loader
import pandas as pd
import os
import plotter
import numpy as np

# # Load kp data:
# kp_data_txt_path = 'C:/Users/sarah.auriemma/Desktop/Data_new/kp_2019.txt'
# dfkp = kp.readKpData(kp_data_txt_path)
# dfkp['Time'] = pd.to_datetime(dfkp['Time'])


sosmag_pickle_dir = 'C:/Users/sarah.auriemma/Desktop/Data_new/6month_study' \
                    '/sosmag'
g17_pickle_dir = 'C:/Users/sarah.auriemma/Desktop/Data_new/6month_study/g17'

# Define the start and end dates for your desired date range
start_date = pd.to_datetime('2019-04-01')
end_date = pd.to_datetime('2019-09-30')

# Load and trim GK2A data
gk2a_time_list, gk2a_data_list, gk2a_89_subtr_list = \
    data_loader.load_and_trim_data(
    sosmag_pickle_dir, start_date, end_date)

# Load and trim GOES-17 data
g17_time_list, g17_data_list, g17_89_subtr_list = \
    data_loader.load_and_trim_data(
    g17_pickle_dir, start_date, end_date)

#
# gk2a_time_list = []
# gk2a_data_list = []
# gk2a_89_subtr_list = []
#
# g17_time_list = []
# g17_data_list = []
# g17_89_subtr_list = []
#
# # Load GK2A data
# for filename in os.listdir(sosmag_pickle_dir):
#     if filename.endswith('.pickle'):
#         file_path = os.path.join(sosmag_pickle_dir, filename)
#         time, ts89_gse, sat_gse = \
#             data_loader.load_model_subtr_gse_from_pickle_file(
#             file_path)
#         gk2a_time_list.extend(time)
#         gk2a_89_subtr_list.extend(ts89_gse)
#         gk2a_data_list.extend(sat_gse)
#
# # Load GOES-17 data
# for filename in os.listdir(g17_pickle_dir):
#     if filename.endswith('.pickle'):
#         file_path = os.path.join(g17_pickle_dir, filename)
#         time, ts89_gse, sat_gse = \
#             data_loader.load_model_subtr_gse_from_pickle_file(
#                 file_path)
#         g17_time_list.extend(time)
#         g17_89_subtr_list.extend(ts89_gse)
#         g17_data_list.extend(sat_gse)
#
# gk2a_time_list = [pd.to_datetime(time) for time in gk2a_time_list]
# g17_time_list = [pd.to_datetime(time) for time in g17_time_list]
###################

# TODO: fix error handling
if len(g17_data_list) == len(gk2a_data_list):
    pass
else:
    print('raise error: len not equal')

# print(len(g17_data_list))

# Get |B| from data using utils
gk2a_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                 for point in gk2a_89_subtr_list]
g17_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                for point in g17_89_subtr_list]

gk2a_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                        in gk2a_data_list]
g17_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                       in g17_data_list]



# plt.plot(gk2a_time_list, gk2a_total_mag_field_modelsub,
#          label='gk2a t89 removed')
# plt.plot(gk2a_time_list, gk2a_total_mag_field, label='gk2a gse')
# plt.legend()
# plt.title('GK2A |B|')
# plt.show()
#
# plt.plot(g17_time_list, g17_total_mag_field_modelsub, label='g17 t89
# removed')
# plt.plot(g17_time_list, g17_total_mag_field, label='g17 gse')
# plt.legend()
# plt.title('G17 |B|')
# plt.show()

# plotter.plot_components_vs_t89('GK2A', gk2a_data_list, gk2a_89_subtr_list)
# plotter.plot_components_vs_t89_with_color('GK2A', gk2a_data_list,
# gk2a_89_subtr_list, gk2a_time_list)


# plotter.plot_components_vs_t89_with_color('G17', g17_data_list,
#                                           g17_89_subtr_list, g17_time_list)
plotter.plot_components_vs_t89_with_color('GK2A', gk2a_data_list,
                                          gk2a_89_subtr_list, gk2a_time_list)
