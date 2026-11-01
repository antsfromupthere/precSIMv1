import numpy as np


def make_pathloss(K, wavelength):

    idx = np.arange(1, K + 1, dtype=float)

    x = 10 * idx
    y = 10 * idx
    location = np.column_stack((x, y))

    d = np.sqrt(
        (10 - 5 * wavelength) ** 2
        + location[:, 0] ** 2 + location[:, 1] ** 2
    )

    pathloss = (wavelength / (4 * np.pi)) ** 2 / (d ** 3.5)

    return location, pathloss
