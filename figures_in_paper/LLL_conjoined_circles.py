# %%
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

from LLL import reduced_circular_coordinate
from circularcoordinates import weighted_circular_coordinate

figures_folder = f'./LLL_figures/'
prime = 47
# %%
data = np.load('counter_LLL_npy/LLL_conjoined_counter.npy')
# %%
plt.figure(figsize=(5, 2.5))
plt.scatter(*data.T)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.tight_layout()
# plt.savefig(f'{figures_folder}counter_dataset.png', dpi=300)
plt.show()
plt.close()
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
distances = ripser_result["dperm2all"]
edges = np.array((distances <= shared_eps).nonzero()).T
# %%
for el1, el2 in zip([orig_0, orig_1], ['counter_orig_0', 'counter_orig_1']):
    plt.figure(figsize=(6, 3))
    plt.scatter(*data.T, c=el1, cmap='hsv')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{figures_folder}conjoined_{el2}.svg')
    plt.close()
# %%
reduced_cc, coeffs, harmonics = reduced_circular_coordinate(data, prime, shared_eps, [0, 1])
new_harmonics = coeffs @ harmonics
# %%
for el1, el2 in zip(reduced_cc, ['counter_new_0', 'counter_new_1']):
    plt.figure(figsize=(6, 3))
    plt.scatter(*data.T, c=el1, cmap='hsv')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{figures_folder}conjoined_{el2}.svg')
    plt.close()

# %%
for i, el in enumerate(harmonics):
    tmp = np.abs(el)
    tmp /= tmp.max()
    plt.figure(figsize=(6, 3))
    plt.scatter(*data.T)
    for el1, el2 in zip(tmp, edges):
        plt.plot(
            *data[el2].T, alpha=el1, c='black', lw=0.5
        )
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{figures_folder}conjoined_counter_orig_harmonic_{i}.svg")
    # plt.show()
    plt.close()
# %%
for i, el in enumerate(new_harmonics):
    tmp = np.abs(el)
    tmp /= tmp.max()
    plt.figure(figsize=(6, 3))
    plt.scatter(*data.T)
    for el1, el2 in zip(tmp, edges):
        plt.plot(
            *data[el2].T, alpha=el1, c='black', lw=0.5
        )
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{figures_folder}conjoined_counter_new_harmonic_{i}.svg")
    # plt.show()
    plt.close()
