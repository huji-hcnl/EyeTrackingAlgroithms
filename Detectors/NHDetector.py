from typing import List
import numpy as np
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from Detectors.BaseDetector import BaseDetector
from scipy.signal import savgol_filter, argrelextrema
from Utils.visual_angle_utils import pixels_array_to_vis_angle_array


class NHDetector(BaseDetector):
    __DEFAULT_WINDOW_LENGTH = 3
    __DEFAULT_POLY_ORDER = 2
    __DEFAULT_ALPHA = 0.7
    __DEFAULT_BETA = 0.3
    __MIN_SACCADE_DURATION = 10  # ms
    __MIN_FIXATION_DURATION = 40  # ms

    def __init__(self, sr: float,
                 pixel_size: float,
                 view_dist: float,
                 timestamps: List[float],
                 window_length: int = __DEFAULT_WINDOW_LENGTH,
                 poly_order: int = __DEFAULT_POLY_ORDER,
                 alpha: float = __DEFAULT_ALPHA,
                 beta: float = __DEFAULT_BETA,
                 min_saccade_duration: float = __MIN_SACCADE_DURATION,
                 min_fixation_duration: float = __MIN_FIXATION_DURATION):
        super().__init__()
        self._pixel_size = pixel_size
        self._view_dist = view_dist
        self._timestamps = timestamps
        self._window_length = window_length
        self._poly_order = poly_order
        self._alpha = alpha
        self._beta = beta
        self._min_saccade_duration = min_saccade_duration
        self._min_fixation_duration = min_fixation_duration

    def _identify_gaze_event_candidates(self, x: np.ndarray,
                                        y: np.ndarray,
                                        candidates: List[GazeEventTypeEnum]) -> List[GazeEventTypeEnum]:
        candidates = np.array(candidates)
        blinks_indexes = np.where(candidates == GazeEventTypeEnum.BLINK)[0]
        # step 1 - filtering, denoising and calculating angular velocities and accelerations
        angular_velocities, angular_accelerations = self._filter_and_denoise(x, y, self._view_dist, self._pixel_size,
                                                                             self._timestamps)
        # step 2- find velocity peaks = local maximums above velocity threshold
        velocity_peak_threshold, mean_of_below_sampled, std_of_below_samples = NHDetector._find_velocity_peaks(
            angular_velocities)
        local_maximum_indexes = argrelextrema(angular_velocities, np.greater)[0]
        velocity_peaks_indexes = [i for i in local_maximum_indexes if angular_velocities[i] > velocity_peak_threshold]

        # step 3- saccade detection
        candidates, saccade_dict, saccade_peaks_list = self._saccade_detection(velocity_peaks_indexes,
                                                                               angular_velocities, self._timestamps,
                                                                               candidates,
                                                                               mean_of_below_sampled,
                                                                               std_of_below_samples)

        # step 4 - glissades detection
        candidates = NHDetector._glissade_detection(angular_velocities, self._timestamps, candidates,
                                                    saccade_peaks_list, saccade_dict)
        # step 5 - fixation detection
        candidates = self._fixation_detection(candidates, self._timestamps)

        # change noise samples to undefined (np.nan in the velocities and accelerations arrays)
        noise_nan_indexes = np.where(np.isnan(angular_velocities) & np.isnan(angular_accelerations))[0]
        for noise_index in noise_nan_indexes:
            candidates[noise_index] = GazeEventTypeEnum.UNDEFINED
        for blink in blinks_indexes:
            candidates[blink] = GazeEventTypeEnum.BLINK

        return list(candidates)

    def _filter_and_denoise(self, x, y, view_dist, pixel_size, timestamps):
        x_filtered = savgol_filter(x, window_length=self._window_length, polyorder=self._poly_order)
        y_filtered = savgol_filter(y, window_length=self._window_length, polyorder=self._poly_order)
        # convert pixels to visual angles and derive
        visual_angles = pixels_array_to_vis_angle_array(x_filtered, y_filtered, view_dist, pixel_size)
        dt = np.concatenate(([np.nan], np.diff(timestamps)))  # first dt is NaN
        angular_velocities = visual_angles / dt
        angular_accelerations = angular_velocities / dt
        # # denoise high values
        angular_velocities[angular_velocities > 1000] = np.nan
        angular_accelerations[angular_accelerations > 100000] = np.nan
        return angular_velocities, angular_accelerations

    def _find_onset_saccade(self, peak_index, angular_velocities, mean_of_below_sampled, std_of_below_samples):
        # saccade will either begin at the beginning of the array, or at the end of another saccade
        current_onset_threshold = 0
        saccade_onset_index = peak_index - 1
        # find local minimum before the peak, calculate its onset threshold and compare to the saccade threshold
        while 0 < saccade_onset_index:
            # if local minimum
            if angular_velocities[saccade_onset_index - 1] > angular_velocities[saccade_onset_index] and \
                    angular_velocities[saccade_onset_index + 1] > angular_velocities[saccade_onset_index]:
                # calculate the current sample's threshold
                current_onset_threshold = mean_of_below_sampled + 3 * std_of_below_samples
                # if local minimum and below threshold = onset
                if angular_velocities[saccade_onset_index] < current_onset_threshold:
                    break
            saccade_onset_index -= 1

        if saccade_onset_index <= 0:
            saccade_onset_index = 0

        return saccade_onset_index, current_onset_threshold

    def _find_offset_saccade(self, peak_index, angular_velocities, timestamps, saccade_onset_threshold):
        saccade_offset_index = peak_index + 1
        saccade_offset_threshold = 0
        # find local minimum after the peak, calculate its offset threshold and compare to the saccade threshold
        while saccade_offset_index < len(angular_velocities) - 1:
            # if local minimum
            if angular_velocities[saccade_offset_index - 1] > angular_velocities[saccade_offset_index] and \
                    angular_velocities[saccade_offset_index + 1] > angular_velocities[saccade_offset_index]:
                # calculate the current sample's threshold in a time window od 40 ms
                window_ending_index = NHDetector._find_window_end_index(saccade_offset_index, 40.0, timestamps)
                velocities_window = angular_velocities[saccade_offset_index: window_ending_index + 1]
                # TODO DECIDE WHICH OPTION
                # option 1 - calculate mean and std on ALL samples in the window
                window_mean = np.mean(velocities_window)
                window_std = np.std(velocities_window)
                saccade_offset_threshold = self._alpha * saccade_onset_threshold + self._beta * \
                                           (window_mean + 3 * window_std)
                if angular_velocities[saccade_offset_index] < saccade_offset_threshold:
                    break

                # option 2 - calculate m & s on window samples that are below the current potential offset
                # below_current_sample = velocities_window[velocities_window < angular_velocities[saccade_offset_index]]
                # # if there is saccade until the end of array, so there aren't samples below the current one
                # if len(below_current_sample) != 0:
                #     saccade_offset_threshold = self._alpha * saccade_onset_threshold + self._beta * \
                #                                (np.mean(below_current_sample) + 3 * np.std(below_current_sample))
                #     # if local minimum and below threshold = offset
                #     if angular_velocities[saccade_offset_index] < saccade_offset_threshold:
                #         break
            saccade_offset_index += 1
        else:
            saccade_offset_index = len(angular_velocities) - 1

        return saccade_offset_index, saccade_offset_threshold

    def _saccade_detection(self, velocity_peaks_indexes, angular_velocities, timestamps, candidates,
                           mean_of_below_sampled, std_of_below_samples):
        # for step 4
        saccade_peaks_list = []  # save all peaks that were recognized as saccades
        saccade_dict = {}  # for each peak save offset threshold and offset index
        for peak_index in velocity_peaks_indexes:
            saccade_onset_index, saccade_onset_threshold = self._find_onset_saccade(peak_index, angular_velocities,
                                                                                    mean_of_below_sampled,
                                                                                    std_of_below_samples)
            saccade_offset_index, saccade_offset_threshold = self._find_offset_saccade(peak_index, angular_velocities,
                                                                                       timestamps,
                                                                                       saccade_onset_threshold)

            if timestamps[saccade_offset_index] - timestamps[saccade_onset_index] > self._min_saccade_duration:
                candidates[saccade_onset_index: saccade_offset_index + 1] = GazeEventTypeEnum.SACCADE
                saccade_peaks_list.append(peak_index)
                saccade_dict[peak_index] = (saccade_offset_threshold, saccade_offset_index, saccade_onset_index)
            else:
                candidates[saccade_onset_index: saccade_offset_index + 1] = GazeEventTypeEnum.UNDEFINED

        return candidates, saccade_dict, saccade_peaks_list

    def _fixation_detection(self, candidates, timestamps):
        # all indexes that aren't saccades or glissades
        # find all undefines sequences, and check if they have a time window of a minimum size of fixation duration
        sequence_start_index, sequence_end_index = 0, 0
        is_fixation_sequence_possible = False
        for i, label in enumerate(candidates):
            if label == GazeEventTypeEnum.UNDEFINED:
                # find the indexes of the undefined sequences, to later check if they are fixations
                if not is_fixation_sequence_possible:
                    is_fixation_sequence_possible = True
                    sequence_start_index, sequence_end_index = i, i
                else:
                    sequence_end_index = i
            # if a sequence ends (a sample that is not undefined) and has a min time duration for fixation
            elif is_fixation_sequence_possible:
                is_fixation_sequence_possible = False
                if timestamps[sequence_end_index] - timestamps[sequence_start_index] > self.__MIN_FIXATION_DURATION:
                    candidates[sequence_start_index: sequence_end_index + 1] = GazeEventTypeEnum.FIXATION

        return candidates

    @staticmethod
    def _find_velocity_peaks(angular_velocities):
        last_PT = 300
        samples_below_last_threshold = angular_velocities[angular_velocities < last_PT]
        mean_of_below_samples = np.mean(samples_below_last_threshold)
        std_of_below_samples = np.std(samples_below_last_threshold)
        current_PT = mean_of_below_samples + 6 * std_of_below_samples
        while np.abs(current_PT - last_PT) >= 1:
            last_PT = current_PT
            samples_below_last_threshold = angular_velocities[angular_velocities < last_PT]
            mean_of_below_samples = np.mean(samples_below_last_threshold)
            std_of_below_samples = np.std(samples_below_last_threshold)
            current_PT = mean_of_below_samples + 6 * std_of_below_samples


        return current_PT, mean_of_below_samples, std_of_below_samples

    @staticmethod
    def _find_window_end_index(window_start_index, time_window_length, timestamps):
        timestamp_temp = np.array(timestamps[window_start_index:])
        timestamp_temp = timestamp_temp - timestamps[window_start_index]
        window_indexes = np.where(timestamp_temp >= time_window_length)[0]
        if window_indexes.size > 0:
            # If there are indexes, return the first one
            window_ending_index = window_indexes[0] + window_start_index
        else:
            # If there are no indexes, return the last index of the array
            window_ending_index = len(timestamps) - 1

        return window_ending_index

    @staticmethod
    def _glissade_detection(angular_velocities, timestamps, candidates, saccade_peaks_list, saccade_dict):
        # for each saccade peak glissade onset will be the saccade's offset
        for i, saccade_peak in enumerate(saccade_peaks_list):
            saccade_offset_threshold = saccade_dict[saccade_peak][0]
            glissade_onset = saccade_dict[saccade_peak][1] + 1
            # check low velocity criteria- samples in time window of 40 ms need to be beneath and above offset threshold
            window_ending_index = NHDetector._find_window_end_index(glissade_onset, 40.0, timestamps)
            velocities_window = angular_velocities[glissade_onset: window_ending_index + 1]
            # the glissade's boundry will be either the next saccade peak or the end of the samples array
            if np.any(velocities_window > saccade_offset_threshold) and \
                    np.any(velocities_window < saccade_offset_threshold):
                if i < len(saccade_peaks_list) - 1:
                    # bound glissade's search to the onset of the next saccade
                    glissade_boundry = saccade_dict[saccade_peaks_list[i + 1]][2]
                    # in case of connected saccades, continue until the last of them
                    if glissade_boundry < saccade_peak:
                        continue
                else:
                    glissade_boundry = len(angular_velocities) - 1
                # the glissade's offset will be the first local minimum below threshold, after the last local maximum
                # that is above the offset threshold
                above_threshold_indexes = np.where(
                    angular_velocities[glissade_onset: glissade_boundry] > saccade_offset_threshold)[0]
                peak_local_max_indexes = argrelextrema(angular_velocities[above_threshold_indexes], np.greater)[0]
                if len(peak_local_max_indexes) != 0:
                    last_max_local = peak_local_max_indexes[-1] + glissade_onset
                    glissade_offset_index = last_max_local + 1
                    while glissade_offset_index < glissade_boundry:
                        if angular_velocities[glissade_offset_index - 1] > angular_velocities[glissade_offset_index] and \
                                angular_velocities[glissade_offset_index + 1] > angular_velocities[glissade_offset_index] \
                                and angular_velocities[glissade_offset_index] < saccade_offset_threshold:
                            break
                        else:
                            glissade_offset_index += 1
                    candidates[glissade_onset: glissade_offset_index + 1] = GazeEventTypeEnum.PSO

        return candidates
