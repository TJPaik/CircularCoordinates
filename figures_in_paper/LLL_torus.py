# %%
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

from LLL import reduced_circular_coordinate
from circularcoordinates import weighted_circular_coordinate

# %%
prime = 47
figures_folder = 'LLL_figures/'
data_coord = np.load('counter_LLL_npy/LLL_torus_counter.npy')
c, a = 4, 2
data = np.asarray([(c + a * np.cos(data_coord[:, 0])) * np.cos(data_coord[:, 1]),
                   (c + a * np.cos(data_coord[:, 0])) * np.sin(data_coord[:, 1]),
                   a * np.sin(data_coord[:, 0])]).T
# %%
ripser_result = ripser(data, coeff=prime, do_cocycles=True)
shared_eps = float(np.mean(np.mean(
    ripser_result['dgms'][1][
        np.argsort(np.diff(ripser_result['dgms'][1], axis=1).flatten())[-2:]
    ], axis=1
)))
# %%
orig_0 = weighted_circular_coordinate(ripser_result, ripser_result=True, prime=prime,
                                      cocycle_n=0, eps=shared_eps)
orig_1 = weighted_circular_coordinate(ripser_result, ripser_result=True, prime=prime,
                                      cocycle_n=1, eps=shared_eps)
original_circular_coordinate = np.asarray([orig_0, orig_1])

# %%
reduced_cc, coeffs, harmonics = reduced_circular_coordinate(data, prime, shared_eps, [0, 1])
# %%
for i, el in zip(range(2), original_circular_coordinate):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*data.T, c=el, cmap='hsv')
    ax.view_init(elev=65, azim=20)
    plt.tight_layout()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    plt.savefig(f'{figures_folder}original_{i}.svg')
    plt.show()
    plt.close()
# %%
for i, el in zip(range(2), reduced_cc):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.scatter(*data.T, c=el, cmap='hsv')
    ax.view_init(elev=65, azim=20)
    plt.tight_layout()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    plt.savefig(f'{figures_folder}correct_{i}.svg')
    plt.show()
    plt.close()
# %%
for el1, el2, el3 in zip(
        [orig_0, orig_0, orig_0, orig_1, orig_1, data_coord[:, 0]],
        [orig_1, data_coord[:, 0], data_coord[:, 1], data_coord[:, 0], data_coord[:, 1], data_coord[:, 1]],
        [
            f'{figures_folder}torus_counter_coord_inf1inf2.svg',
            f'{figures_folder}torus_counter_coord_inf1ori1.svg',
            f'{figures_folder}torus_counter_coord_inf1ori2.svg',
            f'{figures_folder}torus_counter_coord_inf2ori1.svg',
            f'{figures_folder}torus_counter_coord_inf2ori2.svg',
            f'{figures_folder}torus_counter_coord_ori1ori2.svg'

        ]):
    plt.figure(figsize=(3, 3))
    plt.scatter(el1, el2)
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.savefig(el3)
    plt.show()
    plt.close()
    # break

# %%
for el1, el2, el3 in zip(
        [reduced_cc[0], reduced_cc[0], reduced_cc[0], reduced_cc[1], reduced_cc[1], data_coord[:, 0]],
        [reduced_cc[1], data_coord[:, 0], data_coord[:, 1], data_coord[:, 0], data_coord[:, 1], data_coord[:, 1]],
        [
            f'{figures_folder}torus_counter_coord_corrected_inf1inf2.svg',
            f'{figures_folder}torus_counter_coord_corrected_inf1ori1.svg',
            f'{figures_folder}torus_counter_coord_corrected_inf1ori2.svg',
            f'{figures_folder}torus_counter_coord_corrected_inf2ori1.svg',
            f'{figures_folder}torus_counter_coord_corrected_inf2ori2.svg',
            f'{figures_folder}torus_counter_coord_corrected_ori1ori2.svg'

        ]):
    plt.figure(figsize=(3, 3))
    plt.scatter(el1, el2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(el3, dpi=300)
    plt.show()
    plt.close()
# %%
