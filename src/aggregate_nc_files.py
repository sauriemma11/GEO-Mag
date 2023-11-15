import os
import re
from datetime import datetime
import netCDF4 as nc
from ncagg import aggregate, Config
from typing import List


def parse_date(date_str: str) -> datetime:
    """
    Parses a date string in the format YYYYMMDD and returns a datetime object.

    Parameters
    ----------
    date_str (str): A date string in the format 'YYYYMMDD'

    Returns
    -------
    datetime: A datetime object representing the parsed date.

    Raises
    ------
    ValueError: If the date_str is not in the expected format.
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

    Parameters
    ----------
    files : List[str]
        A list of file paths to filter.
    start_date : datetime
        The start of the date range.
    end_date : datetime
        The end of the date range.

    Returns
    -------
    List[str]
        A list of file paths filtered by the specified date range.
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

    Parameters
    ----------
    file_list : List[str]
        A list of NetCDF file paths to aggregate.
    output_file : str
        The path of the output aggregated NetCDF file.

    Raises
    ------
    IOError
        If there is an issue reading the files or writing the output file.

    Raises
    ------
    IOError: If there is an issue reading the file, writing output file,
    or during aggregation process.
    """
    try:
        config = Config.from_nc(file_list[0])
        aggregate(file_list, output_file, config)
    except Exception as e:
        raise IOError(f"Error during file aggregation: {e}")


def aggregate_by_date_range(start_date_str: str, end_date_str: str,
                            directory: str,
                            output_file: str = 'aggregated.nc') -> None:
    """
    Aggregates NetCDF files within a certain time frame specified by the
    start and end dates.

    Parameters
    ----------
    start_date_str : str
        The start date in the format YYYYMMDD.
    end_date_str : str
        The end date in the format YYYYMMDD.
    directory : str
        The directory containing the NetCDF files.
    output_file : str, optional
        The name of the output aggregated NetCDF file, by default
        'aggregated.nc'.

    Raises
    ------
    ValueError
        If the start or end dates are in an invalid format.
    FileNotFoundError
        If no NetCDF files are found in the specified date range or the
        directory cannot be accessed.
    IOError
        If there are issues reading the files or unexpected errors occur.

    """
    if not os.path.isdir(directory):
        raise ValueError(
            f"The specified directory does not exist: {directory}")
    try:
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)

        all_files = [os.path.join(directory, f) for f in os.listdir(directory)
                     if f.endswith('.nc')]
        nc_files = filter_files_by_date(all_files, start_date, end_date)

        if not nc_files:
            raise FileNotFoundError(
                "No NetCDF files found in the specified date range.")

        aggregate_nc_files(nc_files, output_file)
        print(f"Aggregated .nc files into '{output_file}'")

    except ValueError as e:
        raise ValueError(f"Date parsing error: {e}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File error: {e}")
    except Exception as e:
        raise IOError(f"Unexpected error: {e}")

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
