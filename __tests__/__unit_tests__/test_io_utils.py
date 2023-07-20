import os
import unittest

from Utils import io_utils as ioutils


class TestIOUtils(unittest.TestCase):

    def test_create_and_delete_directory(self):
        test_dir = "test_dir"
        test_dirpath = os.path.join(os.getcwd(), test_dir)
        self.assertFalse(os.path.exists(test_dirpath))
        ioutils.create_directory(test_dir, os.getcwd())
        self.assertTrue(os.path.exists(test_dirpath))

        test_file = "test_file.txt"
        test_filepath = os.path.join(test_dirpath, test_file)
        with open(test_filepath, "w") as f:
            f.write("test")
        self.assertTrue(os.path.exists(test_filepath))
        ioutils.delete_directory(test_dirpath)
        self.assertFalse(os.path.exists(test_dirpath))
        self.assertFalse(os.path.exists(test_filepath))
