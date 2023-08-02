import os
import shutil
import zipfile
import pandas as pd
import numpy as np
import requests as requests
from scipy.io import loadmat


# Loads the dataset from article: Andersson, R. et. al (2017): "One algorithm to rule them all? An evaluation and
# discussion of ten eye movement event-detection algorithms".

# constants should be on top of file, e.g:
DATASET_URL = "http://www.kasprowski.pl/datasets/events.zip"
ZIP_NAME = "events.zip"


def load_from_url() -> pd.DataFrame:
    response = requests.get(DATASET_URL)
    with open(ZIP_NAME, "wb") as file:
        file.write(response.content)
    return _extract_and_open_files()


def save_to_pickle(df: pd.DataFrame, path_file: str = "AndersonDataset.pkl") -> None:
    df.to_pickle(path_file)


def load_from_pickle(pkl_path) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    return df


def _extract_file_data(filename):
    dataset = 1
    try:
        rater_name = _extract_rater(filename)
        stimuli = _extract_stimuli(filename)
        return dataset, stimuli, rater_name

    except ValueError as e:
        print(str(e))


def _extract_rater(filename: str) -> str:
    if "_MN" in filename:
        return "MN"
    if "_RA" in filename:
        return "RA"
    raise ValueError("Unknown Rater")


def _extract_stimuli(filename: str) -> str:
    if "img" in filename:
        return "static_image"
    elif "video" in filename:
        return "video_clip"
    else:
        return "moving_dot"


def _open_single_file(file_path) -> pd.DataFrame:
    matlab_file = loadmat(file_path)
    file_data = matlab_file['ETdata']

    # load all data in the file (pos, screenDim, screenRes, viewDist, sampFreq):
    file_type = file_data.dtype
    data_dict = {n: file_data[n][0, 0] for n in file_type.names}
    samples_data = data_dict['pos']
    number_of_samples = samples_data.shape[0]

    # extract file data
    dataset, stimuli, rater_name = _extract_file_data(file_path)
    dataset = [dataset] * number_of_samples
    stimuli = [stimuli] * number_of_samples
    rater_name = [rater_name] * number_of_samples

    # if time samples are not given- complete with difference of 2ms between samples, else divide by 1000 to get ms
    if np.isnan(samples_data[0, 0]):
        samples_data[:, 0] = np.arange(0, samples_data.shape[0]) * 2
    else:
        samples_data[:, 0] -= samples_data[0, 0]
        samples_data[:, 0] /= 1000

    # Create a DataFrame for the current file
    df = pd.DataFrame({"dataset": dataset, "stimuli": stimuli, "rater_name": rater_name, "time": samples_data[:, 0],
                       "right_eye_x": samples_data[:, 3], "right_eye_y": samples_data[:, 4],
                       "label": samples_data[:, 5]})

    return df


def _extract_and_open_files() -> pd.DataFrame:
    # create empty dataframe
    df = pd.DataFrame(columns=["dataset", "stimuli", "rater_name", "time", "right_eye_x", "right_eye_y", "label"])

    # Create a new directory for extracted files
    extracted_directory = "events_extracted_files"
    os.makedirs(extracted_directory, exist_ok=True)

    # Extract the contents of the ZIP file to the new directory
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(extracted_directory)

    internal_dir = "\data"
    full_directory_path = extracted_directory + internal_dir

    # for each file in the directory - process using "open_single_file"
    for filename in os.listdir(full_directory_path):
        file_path = os.path.join(full_directory_path, filename)
        if os.path.isfile(file_path):
            current_file_df = _open_single_file(file_path)
            df = pd.concat([df, current_file_df])

    # delete dataset and new directory
    shutil.rmtree(extracted_directory)
    os.remove(ZIP_NAME)
    return df
