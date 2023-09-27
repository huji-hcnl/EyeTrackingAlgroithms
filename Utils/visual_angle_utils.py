import numpy as np
from typing import Tuple

import constants as cnst
import Utils.array_utils as arr_utils
import Utils.pixel_utils as pixel_utils


def visual_angle_to_pixels(deg: float, d: float, pixel_size: float) -> float:
    """
    Calculates the number of pixels that correspond to a visual angle `deg` degrees, given that the viewer is sitting at
    a distance of `d` centimeters from the screen, and that the size of each pixel is `pixel_size` centimeters.

    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web Book:
        http://www.yorku.ca/eye/visangle.htm

    :param deg: the visual angle (in degrees).
    :param d: the distance (in cm) from the screen.
    :param pixel_size: the size (of the diagonal) of a pixel (in cm).

    :return: the number of pixels that correspond to the given visual angle. If `deg` is not finite, returns np.nan.
    """
    if not np.isfinite(deg):
        return np.nan
    half_edge = d * np.tan(np.deg2rad(abs(deg) / 2))  # in cm
    edge_pixels = 2 * half_edge / pixel_size  # edge size in pixels
    return edge_pixels


def pixels_to_visual_angle(num_px: float, d: float, pixel_size: float, use_radians=False) -> float:
    """
    Calculates the visual angle that corresponds to `num_px` pixels, given that the viewer is sitting at a distance of
    `d` centimeters from the screen, and that the size of each pixel is `pixel_size` centimeters.

    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web Book:
        http://www.yorku.ca/eye/visangle.htm

    :param num_px: the number of pixels.
    :param d: the distance (in cm) from the screen.
    :param pixel_size: the size (of the diagonal) of a pixel (in cm).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the visual angle (in degrees) that corresponds to the given number of pixels.
        If `num_px` is not finite, returns np.nan.

    :raises ValueError: if any of the arguments is negative.
    """
    if not np.isfinite([num_px, d, pixel_size]).all():
        return np.nan
    if (np.array([num_px, d, pixel_size]) < 0).any():
        raise ValueError("arguments `num_px`, `d` and `pixel_size` must be non-negative numbers")
    cm_dist = num_px * pixel_size
    angle = np.arctan(cm_dist / d)
    if not use_radians:
        angle = np.rad2deg(angle)
    return angle


def pixels_array_to_vis_angle_array(xs: np.ndarray, ys: np.ndarray, d: float,
                                    pixel_size: float, use_radians=False) -> np.ndarray:
    """
    Calculates the visual angle between each pair of subsequent pixels in the given x and y coordinates.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param d: distance from the screen in centimeters.
    :param pixel_size: size of each pixel in centimeters.
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.
    :return: visual angle (in degrees) of each point (first is NaN)
    """
    assert len(xs) == len(ys), "x-array and y-array must be of the same length"
    pixel_distances = pixel_utils.calculate_euclidean_distances(xs, ys)
    visual_angles = [pixels_to_visual_angle(px_dist, d, pixel_size, use_radians) for px_dist in pixel_distances]
    return np.array(visual_angles)


def pixels_array_to_vis_angle_velocity_array(xs: np.ndarray, ys: np.ndarray, timestamps: np.ndarray,
                                             d: float, pixel_size: float, use_radians=False) -> np.ndarray:
    """
    Calculates the visual angle between subsequent pixels and divides it by the time difference between the two pixels.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param timestamps: 1D array of timestamps
    :param d: distance from the screen in centimeters.
    :param pixel_size: size of each pixel in centimeters.
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.
    :return: angular velocity (in degrees- or radian-per-second) of each point (first is NaN)
    """
    assert len(xs) == len(ys) == len(timestamps), "x-array, y-array and timestamps-array must be of the same length"
    angles = pixels_array_to_vis_angle_array(xs, ys, d, pixel_size, use_radians)
    cum_angles = np.cumsum(angles)
    return arr_utils.temporal_derivative(cum_angles, timestamps, deg=1, time_coeff=cnst.MILLISECONDS_PER_SECOND)


def visual_angle_between_pixels(p1: Tuple[float, float], p2: Tuple[float, float],
                                distance_from_screen: float, pixel_size: float, use_radians=False) -> float:
    """
    Calculates the visual angle between two pixels on the screen, given that the viewer is sitting at at a distance of
    `distance_from_screen` centimeters from the screen, and that the size of each pixel is `pixel_size` centimeters.
    The returned value is in degrees (or radians if `use_radians` is True).

    :param p1: the (x,y) coordinates of the first pixel.
    :param p2: the (x,y) coordinates of the second pixel.
    :param distance_from_screen: the distance (in cm) from the screen.
    :param pixel_size: the size of each pixel (in cm).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the visual angle between the two pixels (in degrees or radians, depending on `use_radians`). If either
            p1 or p2 is invalid, returns np.nan.
    """
    xs = np.array([p1[0], p2[0]])
    ys = np.array([p1[1], p2[1]])
    if not np.all(np.isfinite(np.concatenate((xs, ys)))):
        # if any of the coordinates is invalid
        return np.nan
    angles = pixels_array_to_vis_angle_array(xs, ys, distance_from_screen, pixel_size, use_radians)
    # angles[0] should be 0, since it's the angle between the first pixel and itself
    assert len(angles) == 2 and angles[0] == 0, "unexpected result from pixels_to_visual_angles"
    return angles[1]
