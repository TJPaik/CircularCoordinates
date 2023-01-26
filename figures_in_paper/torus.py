# %%
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

os.chdir('experiments')
sys.path.append('../')

from circularcoordinates import CircCoordLn, weight_ft_0, weight_ft_with_degree_meta, weighted_circular_coordinate
from dataset import torus

# %%
_ = torus()
data, coord = [_[el] for el in ['data', 'coord']]
prime = 47
labels = ['Original', '1/sqrt(D0 D1)', '1/(D0 + D1)', 'WDGL', ]
coord_1 = coord[:, 0]
coord_2 = coord[:, 1]

# %%
fts = [None] + [
    weight_ft_with_degree_meta(ft)
    for ft in [
        lambda x, y: 1 / (np.sqrt(x * y)),
        lambda x, y: 1 / (x + y)
    ]
] + [
          weight_ft_0(2),
      ]
results = [[
    weighted_circular_coordinate(data, prime=prime, cocycle_n=0, weight_ft=ft),
    weighted_circular_coordinate(data, prime=prime, cocycle_n=1, weight_ft=ft),
] for ft in tqdm(fts)]

# %%
hyper_params = [
    {'epoch': 100, 'lr': 0.001, 'p_norm': 2},
    {'epoch': 3000, 'lr': 0.05, 'p_norm': 4},
    {'epoch': 3000, 'lr': 0.05, 'p_norm': 6},
    {'epoch': 3000, 'lr': 0.05, 'p_norm': 10},
    {'epoch': 5000, 'lr': 0.05, 'p_norm': 20},
]
ccl = CircCoordLn(data, prime=prime)
ccl.setup(0)
orig_1 = ccl.cc_original()
ccl.f_reset_L2()
lp_results_0 = []
for param in hyper_params:
    lp_results_0.append(np.mod(ccl.cc_Lp(**param, delta_thr=1e-7, max_count=500)[0], 1.0))
lp_results_0.append(np.mod(ccl.cc_Linf_Lp(10000, 0.0001, 0.001, 20, 25)[0], 1.0))
lp_results_0 = np.asarray(lp_results_0)

ccl.setup(1)
orig_2 = ccl.cc_original()
ccl.f_reset_L2()
lp_results_1 = []
for param in hyper_params:
    lp_results_1.append(np.mod(ccl.cc_Lp(**param, delta_thr=1e-7, max_count=500)[0], 1.0))
lp_results_1.append(np.mod(ccl.cc_Linf_Lp(10000, 0.0001, 0.001, 20, 25)[0], 1.0))
lp_results_1 = np.asarray(lp_results_1)
lp_results = np.stack([lp_results_0, lp_results_1])

# %%
# joblib.dump([results, lp_results], 'torus_result.pkl')
results, lp_results = joblib.load('torus_result.pkl')
# %%
# original & new
file_name = ['original', 'new']
for i in [0, -1]:
    for j in [0, 1]:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(*data.T, c=results[i][j], cmap='hsv')
        ax.view_init(elev=60, azim=20)
        plt.tight_layout()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        plt.savefig(f'../exp_fig/torus_{file_name[i]}_{j}.svg')
        # plt.show()
        plt.close()
# %%
# scatter plot 0
plt.figure(figsize=(4, 4))
plt.scatter(coord_1, results[0][1], label=labels[0], s=10)
plt.scatter(coord_1, results[-1][1], label=labels[-1], s=10)
plt.scatter(coord_1, lp_results[1][-1], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_0_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
plt.scatter(coord_2, results[0][0], label=labels[0], s=10)
plt.scatter(coord_2, results[-1][0], label=labels[-1], s=10)
plt.scatter(coord_2, lp_results[0][-1], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_0_1.svg')
plt.close()
# %%
# weighted scatter
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord_1, result[1], label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_1_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord_2, result[0], label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_1_1.svg')
plt.close()

# %%
# L\infty sactter
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results[1], [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord_1, result, label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_2_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results[0], [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord_2, result, label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/torus_scatter_2_1.svg')
plt.close()
# %%
