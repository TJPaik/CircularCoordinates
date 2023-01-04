import os

import numpy as np

def noisy_circle(remove=False, location='./circle.npz'):
    if os.path.exists(location):
        if remove:
            os.remove(location)
            return noisy_circle()
        else:
            return np.load(location)
    else:
        n = 300
        coord = np.random.normal(loc=np.pi + 100, scale=0.4 * np.pi, size=n) % (2 * np.pi)
        noise_r = np.random.normal(1, 0.07, n)
        data = (np.asarray([
            np.cos(coord), np.sin(coord)
        ]) * noise_r).T
        np.savez(location, data=data, coord=coord)
        return np.load(location)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = noisy_circle(False)
    plt.scatter(*data['data'].T)
    plt.show()
