import numpy as np
import pandas as pd
from typing import final

from GazeEvents.BaseEvent import BaseEvent
import Utils.pixel_utils as pixel_utils


class BaseGazeEvent(BaseEvent):
    """
    Base class for events that contain gaze data (x,y coordinates).
    """

    def __init__(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray, viewer_distance: float):
        if not np.isfinite(viewer_distance) or viewer_distance <= 0:
            raise ValueError("viewer_distance must be a positive finite number")
        if len(timestamps) != len(x) or len(timestamps) != len(y):
            raise ValueError("Arrays of timestamps, x and y must have the same length")
        super().__init__(timestamps=timestamps)
        self._viewer_distance = viewer_distance  # in cm
        self._x = x
        self._y = y
        self._velocities = pixel_utils.calculate_velocities(xs=self._x,
                                                            ys=self._y,
                                                            timestamps=self._timestamps)  # units: px / ms

    @final
    @property
    def max_velocity(self) -> float:
        """ Returns the maximum velocity of the event in pixels per second """
        return float(np.nanmax(self._velocities)) * 1000

    @final
    @property
    def mean_velocity(self) -> float:
        """ Returns the mean velocity of the event in pixels per second """
        return float(np.nanmean(self._velocities)) * 1000

    @final
    def get_velocities(self) -> np.ndarray:
        """
        Returns the velocities of the event in pixels per millisecond
        """
        return self._velocities

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - max_velocity: the maximum velocity of the event in pixels per second
            - mean_velocity: the mean velocity of the event in pixels per second
        """
        series = super().to_series()
        series["max_velocity"] = self.max_velocity
        series["mean_velocity"] = self.mean_velocity
        return series

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if self._viewer_distance != other._viewer_distance:
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        return True

