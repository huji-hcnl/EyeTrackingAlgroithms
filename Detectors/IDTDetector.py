import numpy as np

class IDTDetector:
    # a threshold value that determines the maximum allowable dispersion for a fixation.
    __DEFAULT_DISPERSION_THRESHOLD = 3.5 #px
    __DEFAULT_WINDOW_SIZE = 2 #size of the time window to calculate the mean of the positions (dispersion)

    def __init__(self,
                 dispersion_threshold: float = __DEFAULT_DISPERSION_THRESHOLD,
                 window_dim: float = __DEFAULT_WINDOW_SIZE):
        self._dispersion_threshold = dispersion_threshold
        self._window_dim = window_dim

    def detect(self, x_coords, y_coords) -> np.ndarray:
        # For each sample calculate the dispersion.
        # one way to calculate it: D = [max(x) - min(x)] + [max(y) - min(y)]
        num_samples = len(x_coords)
        labels = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            if self._window_dim <= i < num_samples - self._window_dim:
                x_window = x_coords[i - self._window_dim : i + self._window_dim + 1]
                y_window = y_coords[i - self._window_dim : i + self._window_dim + 1]
                dispersion = (max(x_window) - min(x_window)) + (max(y_window) - min(y_window))
            else:
                dispersion = 0

            if dispersion < self._dispersion_threshold:
                labels[i] = 1
            else:
                labels[i] = 2

        return labels

# if __name__ == "__main__":
#     data = np.array([[1, 2], [1, 2], [1, 3], [5,7], [5,7]])  # Replace with your eye movement data
#     Xs = data[:, 0]
#     Ys = data[:, 1]
#     detector = IDTDetector()
#     print(detector.detect(Xs, Ys))

