import sys
import os

# Get the absolute path to the src directory
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))

# Add the src directory to sys.path
sys.path.append(src_dir)

# Now you can import timestamp_utils
import timestamp_utils
import datetime
import numpy as np


def test_timestamp_constants():
    epoch, seconds, n_seconds_day = tsu.timestamp_constants()
    assert isinstance(epoch, datetime.datetime)
    assert isinstance(seconds, float)
    assert isinstance(n_seconds_day, int)


def test_j2000_to_posix():
    test_input = np.array([0, 1000000])
    result = tsu.j2000_to_posix(test_input)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(test_input)


def test_j2000_to_posix_0d():
    test_input = 1000000
    result = tsu.j2000_to_posix_0d(test_input)
    assert isinstance(result, datetime.datetime)


def test_j2000_1s_timestamps():
    year, month, day, n_reports = 2000, 1, 1, 60
    result = tsu.j2000_1s_timestamps(year, month, day, n_reports)
    assert isinstance(result, np.ndarray)
    assert result.shape == (24, n_reports)


def test_j2000_p1s_timestamps():
    year, month, day, n_reports, samples_per_report = 2000, 1, 1, 60, 10
    result = tsu.j2000_p1s_timestamps(year, month, day, n_reports,
                                      samples_per_report)
    assert isinstance(result, np.ndarray)
    assert result.shape == (24, n_reports, samples_per_report)


def test_create_j2000_timestamps():
    year, month, day, cadence = 2000, 1, 1, 300
    result = tsu.create_j2000_timestamps(year, month, day, cadence)
    assert isinstance(result, np.ndarray)
    assert len(result) == 288


def test_posix_to_j2000():
    year, month, day, hour, minute, second = 2000, 1, 1, 0, 0, 0
    result = tsu.posix_to_j2000(year, month, day, hour, minute, second)
    assert isinstance(result, float)


def test_iso8601_to_datetime():
    test_input = '2000-01-01T00:00:00.0Z'
    dt, j2000_sec = tsu.iso8601_to_datetime(test_input)
    assert isinstance(dt, datetime.datetime)
    assert isinstance(j2000_sec, float)


def test_doy_to_dom():
    year, doy = 2000, 32
    month, dom = tsu.doy_to_dom(year, doy)
    assert month == 2
    assert dom == 1


def test_dom_to_doy():
    year, month, dom = 2000, 2, 1
    doy = tsu.dom_to_doy(year, month, dom)
    assert doy == 32


def test_doy_to_j2000():
    year, doy = 2000, 1
    result = tsu.doy_to_j2000(year, doy)
    assert isinstance(result, float)


def test_doy_to_j2000_day():
    year, doy = 2000, 1
    n_day_j2k, ms_j2k = tsu.doy_to_j2000_day(year, doy)
    assert isinstance(n_day_j2k, np.int64)
    assert isinstance(ms_j2k, float)


def test_j2000_to_doy():
    test_input = 0
    result = tsu.j2000_to_doy(test_input)
    assert isinstance(result, np.int32)


def test_j2000_to_iso8601():
    test_input = 0
    result = tsu.j2000_to_iso8601(test_input)
    assert isinstance(result, str)


# Run all tests
if __name__ == "__main__":
    test_timestamp_constants()
    test_j2000_to_posix()
    test_j2000_to_posix_0d()
    test_j2000_1s_timestamps()
    test_j2000_p1s_timestamps()
    test_create_j2000_timestamps()
    test_posix_to_j2000
