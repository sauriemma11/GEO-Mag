from data_loader import *
from plotter import *
from utils import calculate_time_difference, find_noon_and_midnight_time
import netCDF4 as nc
import argparse


# TODO: add style checks
# TODO: add option to wget data
# TODO: plot sc orbit locations

def main():
    parser = argparse.ArgumentParser()

    # all spacecraft data is optional, and at least one is required.
    group = parser.add_argument_group('Spacecraft Data')
    group.add_argument("--g17-file", help="File path for GOES-17 mag data")
    group.add_argument("--g18-file", help="File path for GOES-18 mag data")
    group.add_argument("--gk2a-file", help="File path for GK2A SOSMAG data")

    # Optional arguments
    parser.add_argument("--save-path", default=None,
                        help="Save path for the figure \n(optional)")

    parser.add_argument("--g17-deg", default=None,
                        help="GOES17 s/c longitude in degrees (WEST), "
                             "ex. 105\n(optional)")
    parser.add_argument("--g18-deg", default=None,
                        help="GOES18 s/c longitude in degrees (WEST), "
                             "ex. 137.0\n(optional)")
    parser.add_argument("--gk2a-deg", default=None,
                        help="GK2A s/c longitude in degrees (EAST), "
                             "ex. 128.2\n(optional)")

    args = parser.parse_args()

    # Needed to initialize:
    goes17_bgse_stacked, goes18_bgse_stacked, gk2a_bgse_stacked, gk2a_VDH, \
        goes17_VDH, goes17_VDH, save_path = [None] * 7

    # Check if at least one spacecraft data is provided
    if not any([args.g17_file, args.g18_file, args.gk2a_file]):
        parser.error('At least one spacecraft data file must be provided.')

    gk2a_noon, gk2a_midnight, g17_noon, g17_midnight, g18_noon, g18_midnight\
        = [
                                                                                   None] * 6

    if args.g17_file:
        goes17coloc_dataset = nc.Dataset(args.g17_file)
        goes17_bgse_stacked = process_goes_dataset(
            goes17coloc_dataset['b_gse'])
        goes_time_fromnc = goes_epoch_to_datetime(
            goes17coloc_dataset['time'][:])
        goes17_VDH = gse_to_vdh(goes17_bgse_stacked, goes_time_fromnc)

    if args.g18_file:
        goes18coloc_dataset = nc.Dataset(args.g18_file)
        goes18_bgse_stacked = process_goes_dataset(
            goes18coloc_dataset['b_gse'])
        goes_time_fromnc = goes_epoch_to_datetime(
            goes18coloc_dataset['time'][:])
        goes18_VDH = gse_to_vdh(goes18_bgse_stacked, goes_time_fromnc)

    if args.gk2a_file:
        gk2a_dataset = nc.Dataset(args.gk2a_file)
        gk2a_bgse_stacked = stack_gk2a_data(gk2a_dataset)
        gk2a_VDH = gse_to_vdh(gk2a_bgse_stacked, goes_time_fromnc)

    date_str = get_date_str_from_goesTime(goes_time_fromnc)

    # Used to plot 'noon' and 'midnight' times (optional arg)
    noonmidnighttimes_dict = {}

    if args.gk2a_deg:
        gk2a_time_diff = calculate_time_difference(float(args.gk2a_deg), 'E')
        gk2a_noon, gk2a_midnight = find_noon_and_midnight_time(gk2a_time_diff,
                                                               date_str,
                                                               gk2a=True)
        noonmidnighttimes_dict['gk2a'] = {'noon': gk2a_noon,
                                          'midnight': gk2a_midnight}

    if args.g17_deg:
        g17_time_diff = calculate_time_difference(float(args.g17_deg))
        g17_noon, g17_midnight = find_noon_and_midnight_time(g17_time_diff,
                                                             date_str)
        noonmidnighttimes_dict['g17'] = {'noon': g17_noon,
                                         'midnight': g17_midnight}
    if args.g18_deg:
        g18_time_diff = calculate_time_difference(float(args.g18_deg))
        g18_noon, g18_midnight = find_noon_and_midnight_time(g18_time_diff,
                                                             date_str)
        noonmidnighttimes_dict['g18'] = {'noon': g18_noon,
                                         'midnight': g18_midnight}

    # Plot B field in GSE coords:
    plot_BGSE_fromdata_ontop(goes_time_fromnc, goes17_bgse_stacked,
                             goes18_bgse_stacked, 'G17', 'G18', 'SOSMAG',
                             gk2a_bgse_stacked, date_str, save_path,
                             noonmidnighttimes_dict)

    # Plot mag incl (theta) over time:
    plot_magnetic_inclination_over_time_3sc(date_str, goes_time_fromnc,
                                            goes17_VDH, goes18_VDH, gk2a_VDH,
                                            save_path, noonmidnighttimes_dict)

    # print(gk2a_noon, gk2a_midnight, g17_noon, g17_midnight, g18_noon,
    # g18_midnight)
    # print(noonmidnighttimes_dict)

if __name__ == "__main__":
    main()
