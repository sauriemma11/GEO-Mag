from data_loader import *
from plotter import *
import netCDF4 as nc

def main():
    goes18coloc_dataset = nc.Dataset(
        'C:/Users/sarah.auriemma/Desktop/Data_new/g18/mag_1m/08/dn_magn-l2-avg1m_g18_d20220815_v2-0-2.nc')
    goes17coloc_dataset = nc.Dataset(
        'C:/Users/sarah.auriemma/Desktop/Data_new/g17/mag_1m/08/dn_magn-l2-avg1m_g17_d20220815_v2-0-2.nc')
    gk2a_dataset = nc.Dataset('Z:/Data/GK2A/SOSMAG_20220815_b_gse.nc')

    goes_time_fromnc = goes_epoch_to_datetime(goes18coloc_dataset['time'][:])
    goes18_bgse_stacked = process_goes_dataset(goes18coloc_dataset['b_gse'])
    goes17_bgse_stacked = process_goes_dataset(goes17coloc_dataset['b_gse'])

    # gk2a_bgse_stacked = np.column_stack((gk2a_dataset['b_xgse'][:], gk2a_dataset['b_ygse'][:], gk2a_dataset['b_zgse'][:]))
    gk2a_bgse_stacked = stack_gk2a_data(gk2a_dataset)

    plot_BGSE_fromdata_ontop(goes17_bgse_stacked, goes18_bgse_stacked, 'G17', 'G18','SOSMAG',gk2a_bgse_stacked)

    goes17_VDH = gse_to_vdh(goes17_bgse_stacked, goes_time_fromnc)
    goes18_VDH = gse_to_vdh(goes18_bgse_stacked, goes_time_fromnc)
    gk2a_VDH = gse_to_vdh(gk2a_bgse_stacked, goes_time_fromnc)

    # date_str = '2022-08-15'
    date_str = get_date_str_from_goesTime(goes_time_fromnc)
    plot_magnetic_inclination_over_time_3sc(goes_time_fromnc, goes17_VDH, goes18_VDH, gk2a_VDH, date_str)


if __name__ == "__main__":
    main()