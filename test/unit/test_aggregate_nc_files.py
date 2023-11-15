import unittest
import sys
import numpy as np
from unittest import mock

sys.path.insert(0, '../../src')  # noqa
from aggregate_nc_files import *


class TestAggregateFunctions(unittest.TestCase):
    def test_parse_date_valid(self):
        result = parse_date('20200101')
        self.assertEqual(result, datetime(2020, 1, 1))

    def test_parse_date_invalid(self):
        with self.assertRaises(ValueError):
            parse_date('33333838294')  # invalid date
        with self.assertRaises(ValueError):
            parse_date('2020 01 01')  # invalid date
        with self.assertRaises(ValueError):
            parse_date('2020 Jan 01')  # invalid date

    def test_filter_files_by_date(self):
        file_names = ['file_20200101.nc', 'file_20200102.nc',
                      'file_20200103.nc', 'file_20200105.nc',
                      'file_20191231.nc', 'file_20200104.nc']
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 4)
        result = filter_files_by_date(file_names, start_date, end_date)
        self.assertEqual(result, ['file_20200101.nc', 'file_20200102.nc',
                                  'file_20200103.nc', 'file_20200104.nc'])

    @mock.patch('aggregate_nc_files.aggregate')
    @mock.patch('aggregate_nc_files.Config')
    def test_aggregate_nc_files(self, mock_config, mock_aggregate):
        file_list = ['file1.nc', 'file2.nc']
        output_file = 'output.nc'
        aggregate_nc_files(file_list, output_file)
        mock_config.from_nc.assert_called_with(file_list[0])
        mock_aggregate.assert_called_with(file_list, output_file,
                                          mock_config.from_nc.return_value)

    @mock.patch('os.path.isdir')
    @mock.patch('os.listdir')
    @mock.patch('aggregate_nc_files.aggregate_nc_files')
    def test_aggregate_by_date_range(self, mock_aggregate_nc_files,
                                     mock_listdir, mock_isdir):
        # Test 'aggregate_by_date_range' for correct file aggregation in a
        # mocked directory with predefined .nc files and date range.
        mock_isdir.return_value = True
        mock_listdir.return_value = ['file_20200101.nc', 'file_20200201.nc']
        aggregate_by_date_range('20200101', '20200201', 'test_directory')
        mock_aggregate_nc_files.assert_called_once()

    def test_aggregate_by_date_range_invalid_dates(self):
        with self.assertRaises(ValueError):
            aggregate_by_date_range('invalid-date', '20200201',
                                    'test_directory')
