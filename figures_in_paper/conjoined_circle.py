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
from dataset import noisy_conjoined

# %%
_ = noisy_conjoined()
data, coord_1, coord_2 = [_[el] for el in ['data', 'coord_1', 'coord_2']]
prime = 47
labels = ['Original', '1/sqrt(D0 D1)', '1/(D0 + D1)', 'WDGL']

# %%
fts = [None] + [
    weight_ft_with_degree_meta(ft)
    for ft in [lambda x, y: 1 / (np.sqrt(x * y)),
               lambda x, y: 1 / (x + y)]
] + [weight_ft_0(2)]
results = [[
    weighted_circular_coordinate(data, prime=prime, cocycle_n=0, weight_ft=ft),
    weighted_circular_coordinate(data, prime=prime, cocycle_n=1, weight_ft=ft),
] for ft in tqdm(fts)]

# %%
hyper_params = [
    {'epoch': 100, 'lr': 0.001, 'p_norm': 2},
    {'epoch': 2000, 'lr': 0.05, 'p_norm': 4},
    {'epoch': 2000, 'lr': 0.05, 'p_norm': 6},
    {'epoch': 2000, 'lr': 0.05, 'p_norm': 10},
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
# joblib.dump([results, lp_results], 'conjoined_circle_result.pkl')
results, lp_results = joblib.load('conjoined_circle_result.pkl')
# %%
# original
plt.figure(figsize=(8, 4))
plt.scatter(*data.T, c=results[0][0], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_original_1.svg')
plt.close()
plt.figure(figsize=(8, 4))
plt.scatter(*data.T, c=results[0][1], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_original_2.svg')
plt.close()
# %%
# scatter plot 0
plt.figure(figsize=(4, 4))
plt.scatter(coord_1, results[0][1][:300], label=labels[0], s=10)
plt.scatter(coord_1, results[-1][1][:300], label=labels[-1], s=10)
plt.scatter(coord_1, lp_results[1][-1][:300], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_0_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
plt.scatter(coord_2, results[0][0][300:], label=labels[0], s=10)
plt.scatter(coord_2, results[-1][0][300:], label=labels[-1], s=10)
plt.scatter(coord_2, lp_results[0][-1][300:], label='$L^\infty$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_0_1.svg')
plt.close()
# %%
plt.figure(figsize=(8, 4))
plt.scatter(*data.T, c=results[-1][0], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_new_1.svg')
plt.close()
plt.figure(figsize=(8, 4))
plt.scatter(*data.T, c=results[-1][1], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_new_2.svg')
plt.close()
# %%
# weighted scatter
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord_1, result[1][:300], label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_1_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
for result, label in zip(results, labels):
    plt.scatter(coord_2, result[0][300:], label=label, s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_1_1.svg')
plt.close()

# %%
# L\infty sactter
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results[1], [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord_1, result[:300], label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_2_0.svg')
plt.close()
plt.figure(figsize=(4, 4))
for result, label in zip(lp_results[0], [2, 4, 6, '{10}', '{20}', '\infty']):
    plt.scatter(coord_2, result[300:], label=f'$L^{label}$', s=10)
plt.legend()
plt.tight_layout()
plt.savefig('../exp_fig/conjoined_scatter_2_1.svg')
plt.close()
# %%
