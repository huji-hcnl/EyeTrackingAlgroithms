import numpy as np
import Utils.pixel_utils as pixel_utils


def pixels_to_visual_angles(xs: np.ndarray, ys: np.ndarray, d: float, pixel_size: float,
                            use_radians=False) -> np.ndarray:
    """
    Calculates the visual angle of each point in the given x and y coordinates.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param d: distance from the screen in centimeters.
    :param pixel_size: size of each pixel in centimeters.
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.
    :return: visual angle (in degrees) of each point (first is NaN)
    """
    assert len(xs) == len(ys), "x-array and y-array must be of the same length"
    pixel_distances = pixel_utils.calculate_euclidean_distances(xs, ys)
    cm_distances = pixel_distances * pixel_size
    angles = np.arctan(cm_distances / d)
    if not use_radians:
        angles = np.rad2deg(angles)
    return angles


def pixels_to_angular_velocities(xs: np.ndarray, ys: np.ndarray, timestamps: np.ndarray, d: float, pixel_size: float,
                                 use_radians=False) -> np.ndarray:
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
    angles = pixels_to_visual_angles(xs, ys, d, pixel_size, use_radians)
    dt = np.concatenate(([np.nan], np.diff(timestamps)))  # first dt is NaN
    angular_velocities = angles / dt
    return angular_velocities
