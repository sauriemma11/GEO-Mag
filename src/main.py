from data_loader import *
from plotter import *
import netCDF4 as nc
import argparse

def main():
    parser = argparse.ArgumentParser()

    # all spacecraft data is optional, and at least one is required.
    group = parser.add_argument_group('Spacecraft Data')
    group.add_argument("--goes17-file", help="File path for GOES-17 data")
    group.add_argument("--goes18-file", help="File path for GOES-18 data")
    group.add_argument("--sosmag-file", help="File path for SOSMAG data")

    # Optional arguments
    parser.add_argument("--save-path", default=None,
                        help="Save path for the figure (optional)")

    args = parser.parse_args()

    # Needed to initialize:
    goes17_bgse_stacked, goes18_bgse_stacked, gk2a_bgse_stacked, gk2a_VDH, goes17_VDH, goes17_VDH, save_path = [None] * 7

    # Check if at least one spacecraft data is provided
    if not any([args.goes17_file, args.goes18_file, args.sosmag_file]):
        parser.error('At least one spacecraft data file must be provided.')

    if args.goes17_file:
        goes17coloc_dataset = nc.Dataset(args.goes17_file)
        goes17_bgse_stacked = process_goes_dataset(
            goes17coloc_dataset['b_gse'])
        goes_time_fromnc = goes_epoch_to_datetime(
            goes17coloc_dataset['time'][:])
        goes17_VDH = gse_to_vdh(goes17_bgse_stacked, goes_time_fromnc)

    if args.goes18_file:
        goes18coloc_dataset = nc.Dataset(args.goes18_file)
        goes18_bgse_stacked = process_goes_dataset(goes18coloc_dataset['b_gse'])
        goes_time_fromnc = goes_epoch_to_datetime(
            goes18coloc_dataset['time'][:])
        goes18_VDH = gse_to_vdh(goes18_bgse_stacked, goes_time_fromnc)

    if args.sosmag_file:
        gk2a_dataset = nc.Dataset(args.sosmag_file)
        gk2a_bgse_stacked = stack_gk2a_data(gk2a_dataset)
        gk2a_VDH = gse_to_vdh(gk2a_bgse_stacked, goes_time_fromnc)

    date_str = get_date_str_from_goesTime(goes_time_fromnc)

    # Plot B field in GSE coords:
    plot_BGSE_fromdata_ontop(goes_time_fromnc, goes17_bgse_stacked, goes18_bgse_stacked, 'G17', 'G18','SOSMAG',gk2a_bgse_stacked, date_str, save_path)

    # Plot mag incl (theta) over time:
    plot_magnetic_inclination_over_time_3sc(date_str,goes_time_fromnc,goes17_VDH,goes18_VDH,gk2a_VDH,save_path)

if __name__ == "__main__":
    main()