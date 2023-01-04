import os

import numpy as np


def noisy_conjoined(remove=False, location='./conjoined.npz'):
    if os.path.exists(location):
        if remove:
            os.remove(location)
            return noisy_conjoined()
        else:
            return np.load(location)
    else:
        n = 300
        coord_1 = np.random.normal(
            loc=np.pi + 100 + np.random.uniform(0, 2 * np.pi, (1,))[0],
            scale=0.4 * np.pi,
            size=n
        ) % (2 * np.pi)
        coord_2 = np.random.normal(
            loc=np.pi + 100 + np.random.uniform(0, 2 * np.pi, (1,))[0],
            scale=0.4 * np.pi,
            size=n
        ) % (2 * np.pi)
        noise_r = np.random.normal(1, 0.07, n)
        data = (np.asarray([
            np.cos(coord_1), np.sin(coord_1)
        ]) * noise_r).T
        noise_r = np.random.normal(1, 0.07, n)
        data2 = (np.asarray([
            np.cos(coord_2), np.sin(coord_2)
        ]) * noise_r).T + (2, 0)
        data = np.concatenate((data, data2), axis=0)
        np.savez(location, data=data, coord_1=coord_1, coord_2=coord_2)

        return np.load(location)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = noisy_conjoined(False)
    plt.scatter(*data['data'].T)
    plt.show()
