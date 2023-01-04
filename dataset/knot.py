import os

import numpy as np



def noisy_knot(remove=False, location = './knot.npz'
):
    if os.path.exists(location):
        if remove:
            os.remove(location)
            return noisy_knot()
        else:
            return np.load(location)
    else:
        n = 900
        coord = np.mod(np.random.normal(loc=0.5, scale=0.2, size=n),1.0) *(2 * np.pi)
        data = np.asarray([
            1 * np.cos(coord) + 2 * np.cos(2 * coord),
            1 * np.sin(coord) - 2 * np.sin(2 * coord),
            2 * np.sin(3 * coord)
        ]).T
        data += np.random.normal(0, 0.04, (n, 3))
        np.savez(location, data=data, coord=coord)
        return np.load(location)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*noisy_knot(True)['data'].T)
    ax.view_init(elev=80, azim=20)
    plt.tight_layout()
    plt.show()
    plt.close()
