import unittest
from datetime import datetime
import sys

sys.path.insert(0, '../../src')  # noqa
from utils import *


class TestUtils(unittest.TestCase):

    def test_convert_timestamps_to_numeric_valid(self):
        # Test with a valid list of datetime objs
        timestamps = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        result = convert_timestamps_to_numeric(timestamps)
        self.assertEqual(result, [0, 86400])  # 86400 seconds in a day

    def test_convert_timestamps_to_numeric_invalid(self):
        # Test with a list containing non-datetime objs
        timestamps = ['2020-01-01', '2020-01-02']
        with self.assertRaises(TypeError):
            convert_timestamps_to_numeric(timestamps)

    def test_calc_line_of_best_fit_valid(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        result = calc_line_of_best_fit(x, y)
        slope, intercept = result.coefficients
        self.assertAlmostEqual(slope, 2, places=7)  # test slope
        self.assertAlmostEqual(intercept, 0, places=7)  # test intercept

    def test_calc_line_of_best_fit_invalid(self):
        x = [1, 2, 3, 4]
        y = 'not a list'
        with self.assertRaises(TypeError):
            calc_line_of_best_fit(x, y)

    def test_get_avg_data_over_interval_valid(self):
        times = [datetime(2020, 1, 1, i) for i in range(4)]
        data = [1, 2, 3, 4]
        result_times, result_data = get_avg_data_over_interval(times, data,
                                                               '2H')
        self.assertEqual(len(result_times), 2)
        self.assertEqual(result_data, [1.5, 3.5])

    def test_get_avg_data_over_interval_invalid(self):
        times = 'not a list'
        data = [1, 2, 3, 4]
        with self.assertRaises(TypeError):
            get_avg_data_over_interval(times, data)

    def test_find_noon_and_midnight_time_valid(self):
        time_diff = 2  # Hours
        date_str = '2020-01-01'
        noon, midnight = find_noon_and_midnight_time(time_diff, date_str)
        self.assertEqual(noon.hour, 14)
        self.assertEqual(midnight.hour, 2)

    def test_find_noon_and_midnight_time_invalid_date_format(self):
        with self.assertRaises(ValueError):
            find_noon_and_midnight_time(2, '01-01-2020')

    def test_mean_and_std_dev_valid(self):
        data_1 = [1, 2, 3]
        data_2 = [1, 2, 3]
        mean, std_dev = mean_and_std_dev(data_1, data_2)
        self.assertEqual(mean, 0)
        self.assertEqual(std_dev, 0)

    def test_mean_and_std_dev_invalid_length(self):
        data_1 = [1, 2, 3]
        data_2 = [1, 2]
        with self.assertRaises(ValueError):
            mean_and_std_dev(data_1, data_2)


if __name__ == '__main__':
    unittest.main()
