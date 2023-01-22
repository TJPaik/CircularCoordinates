import os

import numpy as np


def torus(remove=False, location='./torus.npz'):
    if os.path.exists(location):
        if remove:
            os.remove(location)
            return torus()
        else:
            return np.load(location)
    else:
        N = 800
        N //= 2
        std = np.pi * 0.4
        coord = np.mod(np.concatenate([
            np.random.normal(0, std, [N, 2]) + (0, np.pi),
            np.random.normal(0, std, [N, 2]) + (np.pi, 0)
        ]), np.pi * 2)
        # coord[:, 0] : meridian
        # coord[:, 1] : longitude
        c, a = 4, 2
        data = np.asarray([(c + a * np.cos(coord[:, 0])) * np.cos(coord[:, 1]),
                           (c + a * np.cos(coord[:, 0])) * np.sin(coord[:, 1]),
                           a * np.sin(coord[:, 0])]).T

        np.savez(location, data=data, coord=coord)
        return np.load(location)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*torus(False)['data'].T)
    ax.view_init(elev=60, azim=20)
    plt.tight_layout()
    plt.show()
    plt.close()
