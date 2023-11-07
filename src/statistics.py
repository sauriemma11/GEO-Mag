import matplotlib.pyplot as plt
import utils
import data_loader
import kp_data_processing as kp
import pandas as pd
import os
import datetime as dtm
import plotter
import numpy as np

# Load kp data:
kp_data_txt_path = 'C:/Users/sarah.auriemma/Desktop/Data_new/kp_2019.txt'
dfkp = kp.readKpData(kp_data_txt_path)
dfkp['Time'] = pd.to_datetime(dfkp['Time'])


sosmag_pickle_dir = 'C:/Users/sarah.auriemma/Desktop/Data_new/6month_study' \
                    '/sosmag'
g17_pickle_dir = 'C:/Users/sarah.auriemma/Desktop/Data_new/6month_study/g17'
# g18_pickle_dir = 'Z:/Data/GOES18/model_outs'

gk2a_time_list = []
gk2a_data_list = []
gk2a_89_subtr_list = []

g17_time_list = []
g17_data_list = []
g17_89_subtr_list = []

# Load GK2A data
for filename in os.listdir(sosmag_pickle_dir):
    if filename.endswith('.pickle'):
        file_path = os.path.join(sosmag_pickle_dir, filename)
        time, ts89_gse, sat_gse = \
            data_loader.load_model_subtr_gse_from_pickle_file(
            file_path)
        gk2a_time_list.extend(time)
        gk2a_89_subtr_list.extend(ts89_gse)
        gk2a_data_list.extend(sat_gse)

# Load GOES-17 data
for filename in os.listdir(g17_pickle_dir):
    if filename.endswith('.pickle'):
        file_path = os.path.join(g17_pickle_dir, filename)
        time, ts89_gse, sat_gse = \
            data_loader.load_model_subtr_gse_from_pickle_file(
                file_path)
        g17_time_list.extend(time)
        g17_89_subtr_list.extend(ts89_gse)
        g17_data_list.extend(sat_gse)

gk2a_time_list = [pd.to_datetime(time) for time in gk2a_time_list]
g17_time_list = [pd.to_datetime(time) for time in g17_time_list]

# Call the function to get the averaged lists
# gk2a_avg_time_list, gk2a_avg_data_list = utils.get_avg_data_over_interval(
#     gk2a_time_list, gk2a_data_list)
# g17_avg_time_list, g17_avg_data_list = utils.get_avg_data_over_interval(
#     g17_time_list, g17_data_list)

# If you need the subtraction lists averaged as well
# gk2a_avg_subtr_time_list, gk2a_avg_subtr_list = \
#     utils.get_avg_data_over_interval(
#     gk2a_time_list, gk2a_89_subtr_list)
# g17_avg_subtr_time_list, g17_avg_subtr_list =
# utils.get_avg_data_over_interval(
#     g17_time_list, g17_89_subtr_list)

# TODO: fix error handling
if len(g17_data_list) == len(gk2a_data_list):
    pass
else:
    print('raise error: len not equal')

# TODO: kp mask isn't working for g17?
# apply kp mask to pickle data, filter out bad values
# kp_mask = kp.createkpMask(dfkp, gk2a_time_list)
# filtered_gk2a_subtr_list = kp.set_subtr_to_nan_where_kp_over(
# gk2a_89_subtr_list, kp_mask)
# filtered_gk2a_datetime_list = kp.set_subtr_to_nan_where_kp_over(
# gk2a_time_list, kp_mask)
# kp_mask = kp.createkpMask(dfkp, g17_time_list)
# filtered_g17_subtr_list = kp.set_subtr_to_nan_where_kp_over(
# g17_89_subtr_list, kp_mask)
# filtered_g17_datetime_list = kp.set_subtr_to_nan_where_kp_over(
# g17_time_list, kp_mask)

# # prep data: No need? Data already prepped when pickle file was made...
# g17_89_daa_array = np.array(filtered_g17_subtr_list)
# g17_89_data_list = data_loader.process_goes_dataset(g17_89_data_array)

gk2a_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                 for point in gk2a_89_subtr_list]
g17_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                for point in g17_89_subtr_list]

gk2a_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                        in gk2a_data_list]
g17_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                       in g17_data_list]

# plotter.plot_4_scatter_plots_with_best_fit(g17_total_mag_field,
# g17_total_mag_field_modelsub, gk2a_total_mag_field,
# gk2a_total_mag_field_modelsub)
#
# plotter.plot_4_scatter_plots_with_color(g17_total_mag_field,
# g17_total_mag_field_modelsub, g17_time_list, gk2a_total_mag_field,
# gk2a_total_mag_field_modelsub, gk2a_time_list)

# plotter.plot_sc_vs_sc_scatter(gk2a_total_mag_field_modelsub,
#                               g17_total_mag_field_modelsub, 'GK2A |B| (GSE)',
#                               'G17 |B| (GSE)', 'GK2A vs G17 GSE (T89
#                               removed)',
#                               lineofbestfit=True)
#
# plotter.plot_sc_vs_sc_scatter(gk2a_total_mag_field, g17_total_mag_field,
#                               'GK2A |B| (GSE)', 'G17 |B| (GSE)',
#                               'GK2A vs G17 GSE', lineofbestfit=True)

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
plotter.plot_components_vs_t89_with_color('G17', g17_data_list,
                                          g17_89_subtr_list, g17_time_list)
