import pickle
import spacepy.plot as splot
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import spacepy.time as sptime
import spacepy.coordinates as spcoords
from datetime import datetime
import spacepy.time as spt
import pandas as pd
### This is all needed for spacepy to work:
import os
if not "CDF_LIB" in os.environ:
    base_dir = "C:/Scripts/cdf3.9.0"
    os.environ["CDF_BASE"] = base_dir
    os.environ["CDF_BIN"] = base_dir + "/bin"
    os.environ["CDF_LIB"] = base_dir + "/lib"
from spacepy import pycdf

def calculate_total_magnetic_field(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def calculate_time_difference(longitude_degrees, hemisphere='W'):
    # Calculate the time difference for the input longitude
    # Input should be in degrees WEST. If east, (GK2A for example is at 128.2 E) input degrees east + 360
    if hemisphere == 'E':
        longitude_degrees = 360 - longitude_degrees

    time_diff = (longitude_degrees / 360) * 24

    return time_diff

def plot_magnetic_field_difference(goes_time, goes_data, gk2a_data, date_str, use_omni, what_model, what_spacecraft, show_figs=True, save_figs=False):
    gk2a_time_diff = calculate_time_difference(128.2, 'E')
    g18_time_diff = calculate_time_difference(137.2)

    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    date_obj_previous_day = date_obj - dt.timedelta(days=1) # For plotting noon time GK2A

    midnight_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0)
    noon_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0)
    noon_time_GK2A = dt.datetime(date_obj_previous_day.year, date_obj_previous_day.month, date_obj_previous_day.day, 12, 0)

    gk2a_midnight_time = midnight_time + dt.timedelta(hours=gk2a_time_diff)
    g18_midnight_time = midnight_time + dt.timedelta(hours=g18_time_diff)
    gk2a_noon_time = noon_time_GK2A + dt.timedelta(hours=gk2a_time_diff)
    g18_noon_time = noon_time + dt.timedelta(hours=g18_time_diff)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

    if use_omni:
        title = f'(SOSMAG - {what_model}), ({what_spacecraft} - {what_model}), using OMNI \n{date_str}'
    else:
        title = f'(SOSMAG - {what_model}), ({what_spacecraft} - {what_model}) \n{date_str}'

    ax1.set_title(title)

    y_annotation = 10

    ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12, annotation_clip=False)
    ax1.annotate('M', xy=(mdates.date2num(g18_midnight_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12, annotation_clip=False)
    ax1.annotate('N', xy=(mdates.date2num(g18_noon_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12, annotation_clip=False)
    ax1.annotate('N', xy=(mdates.date2num(gk2a_noon_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12, annotation_clip=False)

    ax1.plot(goes_time, goes_data[:, 0], 'r')
    ax2.plot(goes_time, goes_data[:, 1], 'r')
    ax3.plot(goes_time, goes_data[:, 2], 'r')

    ax1.plot(goes_time, gk2a_data[:, 0], 'b')
    ax2.plot(goes_time, gk2a_data[:, 1], 'b')
    ax3.plot(goes_time, gk2a_data[:, 2], 'b')

    ax1.legend(['GOES18', 'SOSMAG'])

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    ax1.set_ylabel('B Field GSE, cart. [nT]')
    ax3.set_xlabel('Time [h]')

    plt.tight_layout()

    if show_figs:
        plt.show()

    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}_SOSMAG_{date_str2}_3plts_OMNI.png'
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/{what_spacecraft}/{what_spacecraft}_SOSMAG_{what_model}_{date_str2}_3plts.png'

    if save_figs:
        fig.savefig(filename)

    # Plot total mag field differences
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(goes18_time_1min, subtr)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('|B| [nT]')
    ax1.set_title('Total B field difference for ' + date_str)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show every 2 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.legend().set_visible(False)
    plt.tight_layout()
    if show_figs:
        plt.show()


    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-{what_model}-{what_spacecraft}-{what_model}_totalB_{date_str2}_OMNI.png'
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/{what_spacecraft}/sosmag-{what_model}-{what_spacecraft}-{what_model}_totalB_{date_str2}.png'

    if save_figs:
        fig.savefig(filename)



    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(goes18_time_1min, gk2a_ts04_diff - goes18_ts04_diff)
    date_str = gk2a_time_1min[0].strftime('%Y-%m-%d')

    ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='blue', fontsize=12)
    ax1.annotate(f'M', xy=(mdates.date2num(g18_midnight_time), y_annotation), xytext=(-15, 10),
                 textcoords='offset points', color='red', fontsize=12)


    title = '(SOSMAG - '+whatModel+') - (' + whatSpacecraft + ' - '+whatModel+')\n{}'.format(date_str)
    ax1.set_title(title)
    ax1.set(xlabel='Time [h]', ylabel='B Field GSE [nT]')

    ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1.19, 1), loc='upper right')

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  #show every 2 hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    print(goes18_time_1min[:])
    fig.patch.set_facecolor('white')

    fig.patch.set_alpha(0.6)
    # ax1.grid(False)

    plt.tight_layout()

    if show_figs:
        plt.show()

    if use_omni:
        filename = f'Z:/Data/sos-04-goes-04/{what_spacecraft}/sosmag-{what_model}-{what_spacecraft}-{what_model}_GSE_{date_str2}_OMNI.png'
    else:
        filename = f'Z:/Data/sos-{what_model}-goes-{what_model}/{what_spacecraft}/sosmag-{what_model}-{what_spacecraft}-{what_model}_GSE_{date_str2}.png'

    if save_figs:
        fig.savefig(filename)

def save_subtr_data(datetime, subtr, file_path):
    data_to_save = {'datetime' : datetime, 'subtration' : subtr}
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)

# -----------------------

start_date = dt.date(2022,12,17)
end_date = dt.date(2022,12,17)

whatSpacecraft = 'G18'
whatModel = '89'
# date_str = '2022-12-12' #If doing more than one day we will have to change this

# Flag arguments:
useomni = False # Only can be true if model is 04, otherwise keep False
show_figs = True
save_figs = True
# -------

# Iterate over the date range
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    date_str2 = date_str.replace("-", "")
    print("Processing ", date_str)
    folder_date = current_date.strftime('%Y%m')

    # SOSMAG/GK2A Data --------------
    if useomni == True:
        pickle_path = 'Z:/Data/GK2A/model_outputs/sosmag_modout_OMNI'+date_str+'.pickle'
    else:
        # pickle_path = 'Z:/Data/GK2A/model_outputs/'+date_str2+'/sosmag_modout_'+date_str+'.pickle'
        pickle_path = 'Z:/Data/GK2A/model_outputs/20221217/sosmag_modout_'+date_str+'.pickle'
        # pickle_path = 'Z:/Data/GK2A/model_outputs/sosmag_modout_'+whatModel+date_str+'.pickle'


    gk2a = load_pickle_file(pickle_path)

    attrs = [item for item in gk2a]
    print('Keys for ' + str(pickle_path) + ' : ')
    print(attrs)

    # gk2a_gse, gk2a_time_1min, gk2a_ts04_diff = gk2a['sat_gse'][:1414], gk2a['time_min'][:1414], gk2a['ts04-sat'][:1414]
    gk2a_gse, gk2a_time_1min, gk2a_ts04_diff = gk2a['sat_gse'][:1414], gk2a['time_min'][:1414], gk2a['ts'+whatModel+'-sat'][:1414]

    # GOES18 Data --------------

    ## Aspen EPN data:
    # pickle_path_aspen = 'Z:/Data/GOES18/model_outs/20221217/modout_20221217.pickle'
    # aspen = load_pickle_file(pickle_path_aspen)
    # attrs = [item for item in aspen]
    # print(attrs)
    # plt.plot(aspen['time_min'][:1414], aspen['ts01-sat'][:1414])
    # plt.show()

    if whatSpacecraft == 'G18':
        gpath = 'GOES18'
    else:
        gpath = 'GOES17'

    if useomni == True:
        if whatSpacecraft == 'G18':
            pickle_path = 'Z:/Data/GOES18/model_outs/'+folder_date+'/'+whatSpacecraft+'_modout_OMNI_'+date_str+'.pickle'
        else:
            pickle_path = 'Z:/Data/GOES17/model_outs/' + folder_date + '/' + whatSpacecraft + '_modout_OMNI' + date_str + '.pickle'
    else:
        pickle_path = 'Z:/Data/GOES18/model_outs/20221217/modout_20221217.pickle'
        # pickle_path = 'Z:/Data/'+gpath+'/model_outs/' + folder_date + '/' + whatSpacecraft + '_modout_' + whatModel + date_str + '.pickle'
        # pickle_path = 'Z:/Data/'+gpath+'/model_outs/' + whatSpacecraft + '_modout_' + whatModel + date_str + '.pickle'


    goes18 = load_pickle_file(pickle_path)
    attrs = [item for item in goes18]
    print('Keys for ' + str(pickle_path) + ' : ')
    print(attrs)

    # goes18_gse, goes18_time_1min, goes18_ts04_diff = goes18['sat_gse'][:1414], goes18['time_min'][:1414], goes18['ts04-sat'][:1414]
    # goes18_gse, goes18_time_1min, goes18_ts04_diff = goes18['sat_gse'][:1414], goes18['time_min'][:1414], goes18['ts'+whatModel+'-sat'][:1414]
    goes18_gse, goes18_time_1min, goes18_ts04_diff = goes18['sat'][:1414], goes18['time_min'][:1414], goes18['ts'+whatModel+'-sat'][:1414]



    gk2a_gse_x = gk2a_gse[:,0]
    gk2a_gse_y = gk2a_gse[:,1]
    gk2a_gse_z = gk2a_gse[:,2]

    g18_gse_x = goes18_gse[:,0]
    g18_gse_y = goes18_gse[:,1]
    g18_gse_z = goes18_gse[:,2]

    goes_04_diff_x = goes18_ts04_diff[:,0]
    goes_04_diff_y = goes18_ts04_diff[:,1]
    goes_04_diff_z = goes18_ts04_diff[:,2]

    gk2a_04_diff_x = gk2a_ts04_diff[:,0]
    gk2a_04_diff_y = gk2a_ts04_diff[:,1]
    gk2a_04_diff_z = gk2a_ts04_diff[:,2]

    # Total B field -> SQRT|Bx^2 + By^2 + Bz^2|
    # sosmag_totalB = np.sqrt(gk2a_04_diff_x**2 + gk2a_04_diff_y**2 + gk2a_04_diff_z**2)
    # goes_totalB = np.sqrt(goes_04_diff_x**2 + goes_04_diff_y**2 + goes_04_diff_z**2)


    sosmag_totalB = calculate_total_magnetic_field(gk2a_04_diff_x, gk2a_04_diff_y, gk2a_04_diff_z)
    goes_totalB = calculate_total_magnetic_field(goes_04_diff_x, goes_04_diff_y, goes_04_diff_z)
    # subtr = (sosmag_totalB - goes_totalB)
    subtr = np.abs(sosmag_totalB - goes_totalB)

    # Save subtraction data to pickle file:
    subtraction_file_path = f'Z:/Data/sos-{whatModel}-goes-{whatModel}/{whatSpacecraft}/subtr_pickles/{whatSpacecraft}_{whatModel}subtraction_data_{date_str2}.pickle'

    save_subtr_data(goes18_time_1min, subtr, subtraction_file_path)


    # # Calculate the standard deviation for SOSMAG and GOES
    # # Not sure if this is the right way of doing it...
    # sosmag_std = np.std(sosmag_totalB)
    # goes_std = np.std(goes_totalB)
    #
    # print(f'Standard Deviation (SOSMAG): {sosmag_std:.2f}')
    # print(f'Standard Deviation (GOES): {goes_std:.2f}')
    #
    # plt.plot(goes18_time_1min, subtr, label='Subtraction')
    # plt.plot(goes18_time_1min, sosmag_totalB, label='SOSMAG |B|')
    # plt.plot(goes18_time_1min, goes_totalB, label='GOES |B|')
    # plt.xlabel('Time')
    # plt.ylabel('|B| [nT]')
    # plt.title('Total B Field for ' + date_str)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    # print(gk2a_gse_x, g18_gse_x)

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(gk2a_time_1min, gk2a_gse_x, label='X')
    # ax1.plot(gk2a_time_1min, gk2a_gse_y, label='Y')
    # ax1.plot(gk2a_time_1min, gk2a_gse_z, label='Z')
    # ax1.set_title('SOSMAG B_GSE')
    # ax2.plot(goes18_time_1min, g18_gse_x, label='X')
    # ax2.plot(goes18_time_1min, g18_gse_y, label='Y')
    # ax2.plot(goes18_time_1min, g18_gse_z, label='Z')
    # ax2.set_title('GOES18 B_GSE')
    # plt.tight_layout()
    # ax1.legend()
    # plt.show()

    # -------------
    ## For some reason the last 25 data points always get messed up.
    # plt.plot(goes18_time_1min, goes18_ts01_diff)
    # plt.show()

    # print(goes18_time_1min[-60:], goes18_ts01_diff[-60:])

    # +/- hours for local midnight for sats at longitude coords:
    # gk2a_time_diff = calculate_time_difference(128.2, 'E')
    # g18_time_diff = calculate_time_difference(137.2)
    #
    # date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d')
    #
    # midnight_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0)
    # noon_time = dt.datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0)
    #
    # gk2a_midnight_time = midnight_time + dt.timedelta(hours=gk2a_time_diff)
    # print("sosmag midnight time: ", gk2a_midnight_time)
    # g18_midnight_time = midnight_time + dt.timedelta(hours=g18_time_diff)
    # print("g18 midnight time: ", g18_midnight_time)
    #
    # gk2a_noon_time = noon_time + dt.timedelta(hours=gk2a_time_diff)
    # print("sosmag noon time: ", gk2a_noon_time)
    # g18_noon_time = noon_time + dt.timedelta(hours=g18_time_diff)
    # print("g18 noon time: ", g18_noon_time)
    #
    #
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)  # Add sharex=True and sharey=True
    # if useomni == True:
    #     title = f'(SOSMAG - {whatModel}), ({whatSpacecraft} - {whatModel}), using OMNI \n{date_str}'
    #
    # else:
    #     title = f'(SOSMAG - {whatModel}), ({whatSpacecraft} - {whatModel}) \n{date_str}'
    #
    # ax1.set_title(title)
    #
    # y_annotation = 10 # Y axis location of annotations
    #
    # # TODO: Fix y annotation offset, use max value or something idk
    # # mark where local noon and local midnight are for both spacecraft:
    #
    # ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation), xytext=(-15, 10),
    #              textcoords='offset points', color='blue', fontsize=12, annotation_clip=False)
    # ax1.annotate('M', xy=(mdates.date2num(g18_midnight_time), y_annotation), xytext=(-15, 10),
    #              textcoords='offset points', color='red', fontsize=12, annotation_clip=False)
    # # ax1.annotate('N', xy=(mdates.date2num(gk2a_noon_time), y_annotation), xytext=(-15, 10),
    # #              textcoords='offset points', color='blue', fontsize=12, annotation_clip=False)
    # ax1.annotate('N', xy=(mdates.date2num(g18_noon_time), y_annotation), xytext=(-15, 10),
    #              textcoords='offset points', color='red', fontsize=12, annotation_clip=False)
    #
    #
    # ax1.plot(goes18_time_1min[:], goes18_ts04_diff[:,0], 'r')
    # ax2.plot(goes18_time_1min[:], goes18_ts04_diff[:,1], 'r')
    # ax3.plot(goes18_time_1min[:], goes18_ts04_diff[:,2], 'r')
    #
    # ax1.plot(goes18_time_1min[:], gk2a_ts04_diff[:,0], 'b')
    # ax2.plot(goes18_time_1min[:], gk2a_ts04_diff[:,1], 'b')
    # ax3.plot(goes18_time_1min[:], gk2a_ts04_diff[:,2], 'b')
    #
    # ax1.legend(['GOES18', 'SOSMAG'])
    # # ax3.set(xlabel='Time [h]', ylabel='B Field GSE, cart. [nT]')
    #
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    #
    # # Optionally, you can set the y-axis label for all three plots
    # ax1.set_ylabel('B Field GSE, cart. [nT]')
    # ax3.set_xlabel('Time [h]')
    #
    # plt.tight_layout()
    # if show_figs == True:
    #     plt.show()
    # else:
    #     pass
    #
    #
    # if useomni == True:
    #     if save_figs == True:
    #         fig.savefig(f'Z:/Data/sos-04-goes-04/{gpath}_SOSMAG_{date_str2}_3plts_OMNI.png')
    #     else: pass
    # else:
    #     if save_figs == True:
    #         fig.savefig(f'Z:/Data/sos-{whatModel}-goes-{whatModel}/{whatSpacecraft}/{whatSpacecraft}_SOSMAG_{whatModel}_{date_str2}_3plts.png')
    #     else: pass
    #
    #
    #
    # # fig, (ax1, ax2) = plt.subplots(2, 1)
    # # ax1.plot(gk2a_time_1min, gk2a_ts04_diff)
    # # ax1.set_title('SOSMAG - 04')
    # # ax2.plot(goes18_time_1min, goes18_ts04_diff)
    # # ax2.set_title('GOES17 - 04')
    # # plt.tight_layout()
    # # plt.show()
    #
    # fig, (ax1) = plt.subplots(1, 1)
    # ax1.plot(goes18_time_1min, gk2a_ts04_diff - goes18_ts04_diff)
    # date_str = gk2a_time_1min[0].strftime('%Y-%m-%d')
    #
    # ax1.annotate('M', xy=(mdates.date2num(gk2a_midnight_time), y_annotation), xytext=(-15, 10),
    #              textcoords='offset points', color='blue', fontsize=12)
    # ax1.annotate(f'M', xy=(mdates.date2num(g18_midnight_time), y_annotation), xytext=(-15, 10),
    #              textcoords='offset points', color='red', fontsize=12)
    #
    #
    # title = '(SOSMAG - '+whatModel+') - (' + whatSpacecraft + ' - '+whatModel+')\n{}'.format(date_str)
    # ax1.set_title(title)
    # ax1.set(xlabel='Time [h]', ylabel='B Field GSE [nT]')
    #
    # ax1.legend(['x', 'y', 'z'], bbox_to_anchor=(1.19, 1), loc='upper right')
    #
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))  #show every 2 hours
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    # print(goes18_time_1min[:])
    # fig.patch.set_facecolor('white')
    #
    # fig.patch.set_alpha(0.6)
    # # ax1.grid(False)
    #
    # plt.tight_layout()
    # if show_figs == True:
    #     plt.show()
    # else:
    #     pass
    #
    # if useomni == True:
    #     if save_figs == True:
    #         fig.savefig('Z:/Data/sos-04-goes-04/sosmag-04-'+whatSpacecraft+'-04_GSE_'+date_str+'_OMNI.png', bbox_inches='tight')
    #         print("Saved")
    #     else: pass
    # else:
    #     if save_figs == True:
    #         fig.savefig(f'Z:/Data/sos-{whatModel}-goes-{whatModel}/{whatSpacecraft}/sosmag-{whatModel}-{whatSpacecraft}-{whatModel}_GSE_{date_str}.png', bbox_inches='tight')
    #         print("saved")
    #     else: pass
    #
    #
    # # # Convert date strings to datetime objects
    # # sdate_s = dt.datetime.strptime(date_str + ' 00:00:00', '%Y-%m-%d %H:%M:%S')
    # # edate_s = dt.datetime.strptime(date_str + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
    # #
    # # total_minutes = len(gk2a_time_1min) # Prob 1440 if 1 day. if more than 1 day, we will have to change this.
    # #
    # # time_range = pd.date_range(start=sdate_s, end=edate_s, periods=total_minutes)
    # # time_list = time_range.to_pydatetime().tolist()
    # # tickz = spt.Ticktock(time_list, 'UTC')
    # #
    # # # goes18_ts04_diff_coords = spcoords.Coords(goes18_ts04_diff, 'EPN', 'car', ticks=tickz)
    # # # print(goes18_ts04_diff_coords)
    # #


    plot_magnetic_field_difference(goes18_time_1min, goes18_ts04_diff, gk2a_ts04_diff, date_str, use_omni=False, what_model=whatModel, what_spacecraft=whatSpacecraft, show_figs=show_figs, save_figs=save_figs)

    print(pickle_path)
    current_date += dt.timedelta(days=1)


