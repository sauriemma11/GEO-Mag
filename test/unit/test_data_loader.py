import unittest
import sys
import numpy as np

sys.path.insert(0, '../../src')  #noqa
from data_loader import *


class TestDataLoaderFunctions(unittest.TestCase):
    def test_process_goes_dataset(self):
        test_dataset = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        processed_data = process_goes_dataset(test_dataset)
        expected_result = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertTrue(np.array_equal(processed_data, expected_result))

    def test_process_goes_dataset_randomness(self):
        np.random.seed(0)
        rand_data = np.random.rand(3, 3)
        rand_data[0] = -9999.0
        processed_data = process_goes_dataset(rand_data)

        # make sure correct values were replaced by nans
        self.assertTrue(np.all(np.isnan(processed_data[0])))
        # Check for shape:
        self.assertEqual(processed_data.shape, rand_data.shape)


    def test_process_goes_dataset_withnans(self):
        test_dataset = np.array(
            [[1.0, 2.0, -9999.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        processed_data = process_goes_dataset(test_dataset)
        expected_result = np.array(
            [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertTrue(np.allclose(processed_data, expected_result, equal_nan=True))

    def test_stack_from_data(self):
        # For GOES data
        test_gk2a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        stacked_data = stack_from_data(test_gk2a_data)
        expected_result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertTrue(np.array_equal(stacked_data, expected_result))

    def test_stack_from_data_randomness(self):
        test_sc_data = np.random.rand(3,3)
        stacked_data = stack_from_data(test_sc_data)
        self.assertEqual(stacked_data.shape, (3,3))

    def test_get_date_str_from_goesTime(self):
        # Make sure function is grabbing timestamp correctly from data
        test_goes_time = np.array([pd.to_datetime('2023-10-23'), pd.to_datetime('2023-10-24')])
        date_str = get_date_str_from_goesTime(test_goes_time)
        self.assertEqual(date_str, '2023-10-23')

    def test_stack_gk2a_data(self):
        test_gk2a_data = {
            'b_xgse' : np.array([1.0, 2.0, 3.0]),
            'b_ygse': np.array([4.0, 5.0, 6.0]),
            'b_zgse': np.array([7.0, 8.0, 9.0])
        }
        stacked_data = stack_gk2a_data(test_gk2a_data)
        expected_result = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        self.assertTrue(np.array_equal(stacked_data, expected_result))

    def test_goes_epoch_to_datetime(self):
        test_timestamp = np.array([100000000])
        time_asdtm = goes_epoch_to_datetime(test_timestamp)
        expected_result = pd.to_datetime('2003-03-03 21:46:40')
        self.assertEqual(time_asdtm[0], expected_result)

if __name__ == '__main__':
    unittest.main()
