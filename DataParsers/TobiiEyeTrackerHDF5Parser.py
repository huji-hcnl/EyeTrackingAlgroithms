import os
import h5py as h5
import pandas as pd

from DataParsers.BaseEyeTrackerParser import BaseEyeTrackerParser


class TobiiEyeTrackerHDF5Parser(BaseEyeTrackerParser):
    """
    Parses eye-tracking data based on the HDF5 format exported by Tobii eye-tracker and PsychoPy.
    See additional information here: https://psychopy.org/hardware/eyeTracking.html#what-about-the-data
    """
    # TODO: implement this parser

    @staticmethod
    def tobii_hdf5_to_dataframe(input_path: str):
        """
        Reads the raw HDF5 file exported by Tobii eye-tracker and returns a pandas DataFrame.
        See information about the file structure and data format here:
        https://psychopy.org/hardware/eyeTracking.html#what-about-the-data

        :param input_path: path to the HDF5 file
        :return: pandas DataFrame
        :raises FileNotFoundError: if the file does not exist
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'File not found: {input_path}')
        with h5.File(input_path, 'r') as f:
            dataset = f['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']
            colnames = dataset.dtype.names
            data_dict = {col: [] for col in colnames}
            for row in dataset:
                for i, col in enumerate(colnames):
                    data_dict[col].append(row[i])
        return pd.DataFrame(data_dict)

    raise NotImplementedError






