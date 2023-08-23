import numpy as np
from typing import Tuple


class ScreenMonitor:
    """
    Holds information about the computer screen used for the experiments.
    Default values are taken from the experiment_config.py file.
    """

    def __init__(self, width: float, height: float, refresh_rate: float, resolution: Tuple[int, int]):
        self.__width = width                    # width of the screen in cm
        self.__height = height                  # height of the screen in cm
        self.__refresh_rate = refresh_rate      # refresh rate of the screen in Hz
        self.__resolution = resolution          # resolution of the screen in pixels

    @staticmethod
    def from_tobii_default() -> "ScreenMonitor":
        tobii_width, tobii_height = 53.5, 30.0  # cm
        tobii_refresh_rate = 60  # Hz
        tobii_resolution = (1920, 1080)  # pixels
        return ScreenMonitor(width=tobii_width,
                             height=tobii_height,
                             refresh_rate=tobii_refresh_rate,
                             resolution=tobii_resolution)

    @property
    def width(self) -> float:
        # width of the screen in centimeters
        return self.__width

    @property
    def height(self) -> float:
        # height of the screen in centimeters
        return self.__height

    @property
    def refresh_rate(self) -> float:
        # refresh rate of the screen in Hz
        return self.__refresh_rate

    @property
    def resolution(self) -> Tuple[int, int]:
        # resolution of the screen, i.e. number of pixels in width and height
        return self.__resolution

    @property
    def pixel_size(self) -> float:
        """ Returns the approximate length of one pixel in centimeters (assuming square pixels): cm/px """
        return self.calculate_pixel_size(self.width, self.height, self.resolution)

    def pixels_to_centimeters(self, num_pixels: float) -> float:
        """ Converts pixels to centimeters """
        return num_pixels * self.pixel_size

    @staticmethod
    def calculate_pixel_size(width: float, height: float, resolution: Tuple[int, int]) -> float:
        """ Calculates the approximate length of one pixel in centimeters (assuming square pixels): cm/px """
        diagonal_length = np.sqrt(np.power(width, 2) + np.power(height, 2))  # size of diagonal in centimeters
        diagonal_pixels = np.sqrt(np.power(resolution[0], 2) + np.power(resolution[1], 2))  # size of diagonal in pixels
        return diagonal_length / diagonal_pixels

    def centimeters_to_pixels(self, num_centimeters: float) -> float:
        """ Converts centimeters to pixels """
        return num_centimeters / self.pixel_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.resolution[0]}×{self.resolution[1]}@{self.refresh_rate}Hz)"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, ScreenMonitor):
            return False
        if self.width != other.width:
            return False
        if self.height != other.height:
            return False
        if self.refresh_rate != other.refresh_rate:
            return False
        if self.resolution != other.resolution:
            return False
        return True
