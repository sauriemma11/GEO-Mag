import unittest

import numpy as np

from src.data_loader import *

class TestDataLoaderFunctions(unittest.TestCase):
    def test_process_goes_dataset(self):
        test_dataset = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        processed_data = process_goes_dataset(test_dataset)
        self.assertTrue(all(pd.notna(processed_data))) # is the data not all nan values?
        self.assertEqual(processed_data.shape, (3, 3)) # is the stacked data returning the correct shape

    def test_process_goes_dataset_withnans(self):
        test_dataset = [[1.0, 2.0, -9998.0], [4.0, -9999, 6.0], [7.0, 8.0, 9.0]]
        processed_data = process_goes_dataset(test_dataset)
        expected_processed_data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(processed_data, expected_processed_data)

    def test_stack_from_data(self):
        # This is used for gk2a/sosmag data sets.
        test_sc_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        stacked_data = stack_from_data(test_sc_data)
        expected_result = np.stack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertTrue(np.array_equal(stacked_data, expected_result))

    def test_get_date_str_from_goesTime(self):
        test_goes_time = pd.to_datetime('2023-10-23')
        date_str = get_date_str_from_goesTime(test_goes_time)
        self.assertEqual(date_str, '2023-10-23')

if __name__ == '__main__':
    unittest.main()
