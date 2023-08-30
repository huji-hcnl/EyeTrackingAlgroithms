import numpy as np
import pandas as pd
from typing import Tuple

import Config.experiment_config as cnfg
import Utils.pixel_utils as pixel_utils
import Utils.visual_angle_utils as visang_utils
from GazeEvents.BaseGazeEvent import BaseGazeEvent
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class SaccadeEvent(BaseGazeEvent):
    MIN_DURATION = 5
    MAX_DURATION = 500
    _EVENT_TYPE = GazeEventTypeEnum.SACCADE

    @property
    def start_point(self) -> Tuple[float, float]:
        """ returns the saccade's start point as a tuple of the X,Y coordinates """
        return self._x[0], self._y[0]

    @property
    def end_point(self) -> Tuple[float, float]:
        """ returns the saccade's end point as a tuple of the X,Y coordinates """
        return self._x[-1], self._y[-1]

    @property
    def distance(self) -> float:
        """ returns the distance of the saccade in pixels """
        x_start, y_start = self.start_point
        x_end, y_end = self.end_point
        return np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    @property
    def amplitude(self) -> float:
        """ returns the amplitude of the saccade (visual angle) in degrees """
        return visang_utils.visual_angle_between_pixels(p1=self.start_point, p2=self.end_point,
                                                        distance_from_screen=self._viewer_distance,
                                                        pixel_size=cnfg.SCREEN_MONITOR.pixel_size)

    @property
    def azimuth(self) -> float:
        """ returns the azimuth of the saccade in degrees """
        return pixel_utils.calculate_azimuth(p1=self.start_point, p2=self.end_point, use_radians=False)

    def get_outlier_reasons(self):
        reasons = super().get_outlier_reasons()
        # TODO: check min, max velocity
        return reasons

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
            - distance: saccade's distance (in pixels)
            - amplitude: saccade's visual angle (in degrees)
            - azimuth: saccade's azimuth (in degrees)
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["distance"] = self.distance
        series["amplitude"] = self.amplitude
        series["azimuth"] = self.azimuth
        return series
