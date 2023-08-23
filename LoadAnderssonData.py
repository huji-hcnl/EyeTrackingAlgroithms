import os
import shutil
import zipfile
import pandas as pd
import numpy as np
import requests as requests
from scipy.io import loadmat
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


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


def _extract_global_file_data(filename):
    dataset = 1
    try:
        rater_name = _extract_rater(filename)
        stimuli = _extract_stimuli(filename)
        return dataset, stimuli, rater_name

    except ValueError as e:
        print(str(e))


def _extract_from_mat_file(file_data):
    # load all data in the file (pos, screenDim, screenRes, viewDist, sampFreq):
    file_type = file_data.dtype
    data_dict = {n: file_data[n][0, 0] for n in file_type.names}
    samples_data = data_dict['pos']
    view_dist = data_dict['viewDist'] * 100  # change from meters to cm
    screen_dim_width = data_dict['screenDim'][0][0] * 100
    screen_dim_height = data_dict['screenDim'][0][1] * 100
    screen_res_width = data_dict['screenRes'][0][0]
    screen_res_height = data_dict['screenRes'][0][1]
    number_of_samples = samples_data.shape[0]

    # calculate pixel size
    from Config.ScreenMonitor import ScreenMonitor
    pixel_size = ScreenMonitor.calculate_pixel_size(width=screen_dim_width, height=screen_dim_height,
                                                    resolution=(screen_res_width, screen_res_height))

    return samples_data, view_dist, pixel_size, number_of_samples


def _extract_time_stamp(samples_data):
    # if time samples are not given- complete with difference of 2ms between samples, else divide by 1000 to get ms
    if np.isnan(samples_data[0]):
        samples_data[:] = np.arange(0, samples_data.shape[0]) * 2
    else:
        samples_data[:] -= samples_data[0]
        samples_data[:] /= 1000

    return samples_data


def _extract_labels(samples_data):
    enums_dict = {1: GazeEventTypeEnum.FIXATION, 2: GazeEventTypeEnum.SACCADE, 3: GazeEventTypeEnum.PSO,
                  4: GazeEventTypeEnum.SMOOTH_PURSUIT, 5: GazeEventTypeEnum.BLINK, 6: GazeEventTypeEnum.UNDEFINED}

    return list(map(lambda v: enums_dict[v], samples_data))


def _open_single_file(file_path) -> pd.DataFrame:
    matlab_file = loadmat(file_path)
    file_data = matlab_file['ETdata']

    # extract global file data
    samples_data, view_dist, pixel_size, number_of_samples = _extract_from_mat_file(file_data)
    dataset, stimuli, rater_name = _extract_global_file_data(file_path)

    # create columns from global data
    dataset = [dataset] * number_of_samples
    stimuli = [stimuli] * number_of_samples
    rater_name = [rater_name] * number_of_samples
    view_dist = [view_dist] * number_of_samples
    pixel_size = [pixel_size] * number_of_samples
    time_stamp_col = _extract_time_stamp(samples_data[:, 0])
    labels = _extract_labels(samples_data[:, 5])

    # Create a DataFrame for the current file
    df = pd.DataFrame({"dataset": dataset, "stimuli": stimuli, "rater_name": rater_name,
                       "viewer_distance_cm": view_dist, "pixel_size_cm": pixel_size, "time": time_stamp_col,
                       "right_eye_x": samples_data[:, 3], "right_eye_y": samples_data[:, 4],
                       "label": labels})

    return df


def _extract_and_open_files() -> pd.DataFrame:
    # create empty dataframe
    df = pd.DataFrame(columns=["dataset", "stimuli", "rater_name", "viewer_distance_cm", "pixel_size_cm",
                               "time", "right_eye_x", "right_eye_y", "label"])

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
