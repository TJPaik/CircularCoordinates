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
from dataset import noisy_circle

# %%
_ = noisy_circle()
data, coord = [_[el] for el in ['data', 'coord']]
prime = 47
labels = ['Original', '1/sqrt(D0 D1)', '1/(D0 + D1)', 'WDGL']

# %%
fts = [None] + [
    weight_ft_with_degree_meta(ft)
    for ft in [
        lambda x, y: 1 / (np.sqrt(x * y)),
        lambda x, y: 1 / (x + y)
    ]
] + [weight_ft_0(2)]
results = [weighted_circular_coordinate(data, prime=prime, cocycle_n=0, weight_ft=ft) for ft in tqdm(fts)]

# %%
ccl = CircCoordLn(data, prime=prime)
ccl.setup(0)
ccl.cc_original()
ccl.f_reset_L2()
hyper_params = [
    {'epoch': 100, 'lr': 0.001, 'p_norm': 2},
    {'epoch': 1000, 'lr': 0.05, 'p_norm': 4},
    {'epoch': 1000, 'lr': 0.05, 'p_norm': 6},
    {'epoch': 2000, 'lr': 0.05, 'p_norm': 10},
    {'epoch': 2000, 'lr': 0.05, 'p_norm': 20},
]
lp_results = []
for param in hyper_params:
    lp_results.append(np.mod(ccl.cc_Lp(**param)[0], 1.0))
lp_results.append(np.mod(ccl.cc_Linf_Lp(5000, 0.0001, 0.001, 20, 25)[0], 1.0))
lp_results = np.mod(np.asarray(lp_results), 1.0)

# %%
# joblib.dump([results, lp_results], 'circle_result.pkl')
results, lp_results = joblib.load('circle_result.pkl')
# %%
# original
plt.figure(figsize=(4, 4))
plt.scatter(*data.T, c=results[0], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/circle_original.svg')
plt.close()
# %%
# scatter plot 0
plt.figure(figsize=(4, 4))
plt.scatter(coord, results[0], label=labels[0], s=10)
plt.scatter(coord, results[-1], label=labels[-1], s=10)
plt.scatter(coord, lp_results[-1], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/circle_scatter_0.svg')
plt.close()
# %%
plt.figure(figsize=(4, 4))
plt.scatter(*data.T, c=lp_results[-1], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/circle_new.svg')
plt.close()
# %%
# weighted scatter
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord, result, label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/circle_scatter_1.svg')
plt.close()
# %%
# L\infty sactter
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results, [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord, result, label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/circle_scatter_2.svg')
plt.close()
# %%
