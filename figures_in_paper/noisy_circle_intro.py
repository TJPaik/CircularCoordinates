# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

os.chdir('experiments')
sys.path.append('../')

from circularcoordinates import weighted_circular_coordinate
from dataset import noisy_circle

# %%
_ = noisy_circle()
data1, _ = [_[el] for el in ['data', 'coord']]
prime = 47

n = len(data1)
coord = np.random.uniform(0, np.pi * 2, n)
noise_r = np.random.normal(1, 0.07, n)
data2 = (np.asarray([np.cos(coord), np.sin(coord)]) * noise_r).T
# %%
results1 = weighted_circular_coordinate(data1, prime=prime, cocycle_n=0, weight_ft=None)
results2 = weighted_circular_coordinate(data2, prime=prime, cocycle_n=0, weight_ft=None)

# %%
plt.figure(figsize=(4, 4))
plt.scatter(*data1.T, c=results1, cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/circle_intro_density.svg')
plt.close()
plt.figure(figsize=(4, 4))
plt.scatter(*data2.T, c=results2, cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../exp_fig/circle_intro_no_density.svg')
plt.close()
