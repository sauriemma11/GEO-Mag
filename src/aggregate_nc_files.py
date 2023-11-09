import os
import re
from datetime import datetime
import netCDF4 as nc
from ncagg import aggregate, Config
from typing import List


def parse_date(date_str: str) -> datetime:
    """
    Parses a date string in the format YYYYMMDD and returns a datetime object.
    """
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError as e:
        raise ValueError(
            f"Not a valid date: '{date_str}'. Expected format: YYYYMMDD.") \
            from e


def filter_files_by_date(files: List[str], start_date: datetime,
                         end_date: datetime) -> List[str]:
    """
    Filters a list of file paths based on whether their names contain a date
    within the given range.
    """
    date_pattern = re.compile(r'(\d{8})')
    filtered_files = []
    for file_path in files:
        match = date_pattern.search(file_path)
        if match:
            file_date = parse_date(match.group(1))
            if start_date <= file_date <= end_date:
                filtered_files.append(file_path)
    return filtered_files


def aggregate_nc_files(file_list: List[str], output_file: str) -> None:
    """
    Aggregates a list of NetCDF files into a single output file using ncagg.
    """
    config = Config.from_nc(file_list[0])
    aggregate(file_list, output_file, config)


def aggregate_by_date_range(start_date_str: str, end_date_str: str,
                            directory: str,
                            output_file: str = 'aggregated.nc') -> None:
    """
    Aggregates NetCDF files within a certain time frame specified by the
    start and end dates.
    """
    start_date = parse_date(start_date_str)
    end_date = parse_date(end_date_str)

    # List all files in the directory and filter them by date
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                 f.endswith('.nc')]
    nc_files = filter_files_by_date(all_files, start_date, end_date)

    if not nc_files:
        print("No NetCDF files found in the specified date range.")
        return

    aggregate_nc_files(nc_files, output_file)
    print(f"Aggregated .nc files into '{output_file}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate NetCDF files within a certain time frame.")
    parser.add_argument("start_date",
                        help="Start date in the format YYYYMMDD.")
    parser.add_argument("end_date", help="End date in the format YYYYMMDD.")
    parser.add_argument("directory",
                        help="Directory containing the .nc files.")
    parser.add_argument("-o", "--output", default="aggregated.nc",
                        help="Output NetCDF file name.")

    args = parser.parse_args()

    aggregate_by_date_range(args.start_date, args.end_date, args.directory,
                            args.output)
