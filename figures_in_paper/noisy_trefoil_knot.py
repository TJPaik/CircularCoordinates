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
from dataset import noisy_knot

# %%
_ = noisy_knot()
data, coord = [_[el] for el in ['data', 'coord']]
prime = 47
labels = ['Original', '1/sqrt(D0 D1)', '1/(D0 + D1)', 'WDGL']

# %%
fts = [None] + [
    weight_ft_with_degree_meta(ft)
    for ft in [lambda x, y: 1 / (np.sqrt(x * y)),
               lambda x, y: 1 / (x + y)]
] + [weight_ft_0(2)]
results = [weighted_circular_coordinate(data, prime=prime, cocycle_n=0, weight_ft=ft) for ft in tqdm(fts)]

# %%
ccl = CircCoordLn(data, prime=prime)
ccl.setup(0)
ccl.cc_original()
ccl.f_reset_L2()
hyper_params = [
    {'epoch': 7000, 'lr': 0.001, 'p_norm': 2},
    {'epoch': 7000, 'lr': 0.05, 'p_norm': 3},
    {'epoch': 7000, 'lr': 0.05, 'p_norm': 4},
    {'epoch': 7000, 'lr': 0.01, 'p_norm': 5},
    {'epoch': 30000, 'lr': 0.01, 'p_norm': 6},
    {'epoch': 30000, 'lr': 0.01, 'p_norm': 10},
]
lp_results = []
for param in hyper_params:
    lp_results.append(ccl.cc_Lp(**param, delta_thr=1e-7, max_count=500)[0])
lp_results.append(ccl.cc_Linf_Lp(30000, 0.0001, 0.001, 10, 25)[0])
lp_results = np.mod(np.asarray(lp_results), 1.0)

# %%
# joblib.dump([results, lp_results], 'knot_result.pkl')
results, lp_results = joblib.load('knot_result.pkl')
# %%
# original
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
ax.scatter(*data.T, c=results[0], cmap='hsv')
ax.view_init(elev=80, azim=20)
plt.tight_layout()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.savefig('../exp_fig/knot_original.svg')
plt.close()
# %%
# scatter plot 0
plt.figure(figsize=(4, 4))
plt.scatter(coord, results[0], label=labels[0], s=10)
plt.scatter(coord, results[-1], label=labels[-1], s=10)
plt.scatter(coord, lp_results[-1], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/knot_scatter_0.svg')
plt.close()
# %%
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
ax.scatter(*data.T, c=results[-1], cmap='hsv')
ax.view_init(elev=80, azim=20)
plt.tight_layout()
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
plt.savefig('../exp_fig/knot_new.svg')
plt.close()
# %%
# weighted scatter
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord, result, label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/knot_scatter_1.svg')
plt.close()
# %%
# L\infty sactter
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results, [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord, result, label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/knot_scatter_2.svg')
plt.close()
# %%
