import unittest

import numpy as np
import pandas as pd

import Utils.array_utils as arr_utils


class TestIOUtils(unittest.TestCase):

    def test_is_one_dimensional(self):
        self.assertTrue(arr_utils.is_one_dimensional([1, 2, 3]))
        self.assertTrue(arr_utils.is_one_dimensional([[1], [2], [3]]))
        self.assertTrue(arr_utils.is_one_dimensional([[1, 2, 3]]))
        self.assertFalse(arr_utils.is_one_dimensional([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, arr_utils.is_one_dimensional, [[1, 2], [3]])

    def test_extract_column_safe(self):
        data = pd.DataFrame(np.random.rand(10, 5), columns=[f"col{i}" for i in range(5)])

        colname = "col4"
        obs = arr_utils.extract_column_safe(data, colname)
        exp = data[colname].values
        self.assertTrue(np.array_equal(obs, exp))

        colname = "col5"
        obs = arr_utils.extract_column_safe(data, colname, warn=False)
        exp = np.full(shape=data.shape[0], fill_value=np.nan)
        self.assertTrue(np.array_equal(obs, exp, equal_nan=True))

        colname = "col5"
        self.assertWarns(UserWarning, arr_utils.extract_column_safe, data, colname, warn=True)

    def test_get_chunk_indices(self):
        arr = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

        arr[2] = np.nan
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([0, 1]), np.array([2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

        arr = np.arange(-5, 5)
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([i]) for i in range(10)]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

    def test_find_sequences_in_sparse_array(self):
        seq = np.array([1, 2, 3])

        arr = np.array([np.nan, 2, 3, np.nan, 1, 2, np.nan, 3, 2, np.nan, np.nan, 3, 1, np.nan, np.nan, np.nan, 2,
                        np.nan, 3, 1, 1, np.nan])
        exp = [(4, 7), (12, 18)]
        res = arr_utils.find_sequences_in_sparse_array(arr, seq)
        self.assertEqual(exp, res)

        arr = np.array([3, 2, 1, 2, 1])
        exp = []
        res = arr_utils.find_sequences_in_sparse_array(arr, seq)
        self.assertEqual(exp, res)
