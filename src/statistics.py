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
gk2a_time_list, gk2a_data_list, gk2a_89_model_list, gk2a_89_subtr_list = \
    data_loader.load_and_trim_data(
    sosmag_pickle_dir, start_date, end_date)

# Load and trim GOES-17 data
g17_time_list, g17_data_list, g17_89_model_list, g17_89_subtr_list = \
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
gk2a_total_mag_field_model = [utils.calculate_total_magnetic_field(*point)
                              for point in gk2a_89_model_list]
g17_total_mag_field_model = [utils.calculate_total_magnetic_field(*point)
                             for point in g17_89_model_list]

gk2a_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                 for point in gk2a_89_subtr_list]
g17_total_mag_field_modelsub = [utils.calculate_total_magnetic_field(*point)
                                for point in g17_89_subtr_list]

gk2a_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                        in gk2a_data_list]
g17_total_mag_field = [utils.calculate_total_magnetic_field(*point) for point
                       in g17_data_list]

# aligned_g17, aligned_gk2a = utils.align_datasets(g17_time_list,
# gk2a_time_list,
#                                                  g17_total_mag_field_modelsub,
#                                                  gk2a_total_mag_field_modelsub)

# mean_difference_sub, standard_deviation_sub = utils.mean_and_std_dev(
#     aligned_gk2a,
#     aligned_g17)

# aligned_g17, aligned_gk2a = utils.align_datasets(g17_time_list,
# gk2a_time_list,
#                                                  g17_total_mag_field,
#                                                  gk2a_total_mag_field)

mean_difference_g17_obsvsmodel, standard_deviation_g17_obsvsmodel = \
    utils.mean_and_std_dev(
    g17_total_mag_field_model, g17_total_mag_field)
mean_difference_gk2a_obsvsmodel, standard_deviation_gk2a_obsvsmodel = \
    utils.mean_and_std_dev(
    gk2a_total_mag_field_model, gk2a_total_mag_field)

print('g17 |B| (GSE) obsv vs TS89 model')
print('bottom left sub plot')
print(f'Mean Difference: {mean_difference_g17_obsvsmodel} nT')
print(f'Standard Deviation: {standard_deviation_g17_obsvsmodel} nT')
print('---------------')
print('gk2a |B| (GSE) obsv vs TS89 model')
print('bottom right sub plot')
print(f'Mean Difference: {mean_difference_gk2a_obsvsmodel} nT')
print(f'Standard Deviation: {standard_deviation_gk2a_obsvsmodel} nT')
print('---------------')

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


# plotter.plot_components_vs_t89_with_color('G17', g17_data_list,
#                                           g17_89_subtr_list, g17_time_list)
# plotter.plot_components_vs_t89_with_color('GK2A', gk2a_data_list,
#                                           gk2a_89_subtr_list, gk2a_time_list)


plotter.plot_4_scatter_plots_with_color(
    g17_total_mag_field, g17_total_mag_field_model, g17_time_list,
    gk2a_total_mag_field, gk2a_total_mag_field_model, gk2a_time_list,
    output_file=None, best_fit=True, is_model_subtr=False)

# plotter.plot_4_scatter_plots_with_color(
#     g17_total_mag_field, g17_total_mag_field_modelsub, g17_time_list,
#     gk2a_total_mag_field,
#     gk2a_total_mag_field_modelsub, gk2a_time_list, output_file=None,
#     best_fit=True, is_model_subtr=True)


mean_difference_subtrvssubtr, standard_deviation_subtrvssubtr = \
    utils.mean_and_std_dev(
        g17_total_mag_field_modelsub, gk2a_total_mag_field_modelsub)

print('(g17 - T89 model) vs (gk2a - T89 model)')
print('top left plot')
print(f'Mean Difference: {mean_difference_subtrvssubtr} nT')
print(f'Standard Deviation: {standard_deviation_subtrvssubtr} nT')
print('---------------')

# Use the function to calculate stats
# hourly_stats = utils.calc_stats(gk2a_time_list,
# gk2a_total_mag_field_modelsub, g17_total_mag_field_modelsub)
# hourly_stats = utils.calc_stats(gk2a_time_list,
# gk2a_total_mag_field_modelsub, g17_total_mag_field_modelsub, 'H')
# print(hourly_stats)


# To produce a table suitable for a scientific journal, you would likely do
# further formatting:
# table_for_journal = hourly_stats.style.format("{:.2f}").set_table_styles(
#     [{"selector": "th", "props": [("font-size", "10pt")]}]
# )


# TODO: add if __name__ == '__main__' to avoid running on import
