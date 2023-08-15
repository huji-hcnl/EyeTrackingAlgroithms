import numpy as np
import pandas as pd
from typing import Tuple

from GazeEvents.BaseGazeEvent import BaseGazeEvent
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class FixationEvent(BaseGazeEvent):
    MIN_DURATION = 50
    MAX_DURATION = 2000
    _EVENT_TYPE = GazeEventTypeEnum.FIXATION

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, pupil: np.ndarray, viewer_distance: float):
        super().__init__(timestamps=timestamps, x=x, y=y, viewer_distance=viewer_distance)
        self._pupil: np.ndarray = pupil  # pupil size (in mm)

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """ returns the mean coordinates of the fixation on the X,Y axes """
        x_mean = float(np.nanmean(self._x))
        y_mean = float(np.nanmean(self._y))
        return x_mean, y_mean

    @property
    def standard_deviation(self) -> Tuple[float, float]:
        """ returns the standard deviation of the fixation on the X,Y axes """
        x_std = float(np.nanstd(self._x))
        y_std = float(np.nanstd(self._y))
        return x_std, y_std

    @property
    def dispersion(self) -> float:
        """ returns the maximum distance between any two points in the fixation (in pixels units) """
        points = np.column_stack((self._x, self._y))
        distances = np.linalg.norm(points - points[:, None], axis=-1)
        max_dist = float(np.nanmax(distances))
        return max_dist

    @property
    def mean_pupil_size(self) -> float:
        """ returns the mean pupil size during the fixation (in mm) """
        return float(np.nanmean(self._pupil))

    @property
    def std_pupil_size(self) -> float:
        """ returns the standard deviation of the pupil size during the fixation (in mm) """
        return float(np.nanstd(self._pupil))

    def get_outlier_reasons(self):
        reasons = super().get_outlier_reasons()
        # TODO: check max velocity, acceleration, dispersion
        # TODO: check if inside the screen
        return reasons

    def get_pupil_sizes(self) -> np.ndarray:
        """ returns the pupil size during the fixation (in mm) """
        return self._pupil

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of fixation information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - center_of_mass: fixation's center of mass (2D pixel coordinates)
            - standard_deviation: fixation's standard deviation (in pixel units)
            - dispersion: maximum distance between any two points in the fixation (in pixels units)
            - mean_pupil_size: mean pupil size during the fixation (in mm)
            - std_pupil_size: standard deviation of the pupil size during the fixation (in mm)
        """
        series = super().to_series()
        series["center_of_mass"] = self.center_of_mass
        series["standard_deviation"] = self.standard_deviation
        series["dispersion"] = self.dispersion
        series["mean_pupil_size"] = self.mean_pupil_size
        series["std_pupil_size"] = self.std_pupil_size
        return series



