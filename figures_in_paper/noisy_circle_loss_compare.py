import os
import sys

import matplotlib.pyplot as plt
import numpy as np

os.chdir('experiments')
sys.path.append('../')

from circularcoordinates import CircCoordLn
from dataset import noisy_circle

# %%
_ = noisy_circle()
data, coord = [_[el] for el in ['data', 'coord']]
prime = 47

ccl = CircCoordLn(data, prime=prime)
ccl.setup(0)
ccl.cc_original()
hyper_parameters = [
    [0.005, 1e-4],
    [0.001, 1e-4],
    [0.005, 1e-3],
    [0.01, 1e-3],
]
# %%
results = []
epochs = 10000
for lr, delta_thr in hyper_parameters:
    ccl.f_reset()
    a0, b0 = ccl.cc_Lp(epochs, lr=lr, p_norm=np.inf)
    ccl.f_reset()
    a1, b1 = ccl.cc_Linf_Lp(epochs, delta_thr=delta_thr, lr=lr, lower_p=2, upper_p=50)
    ccl.f_reset()
    a2, b2 = ccl.cc_Linf_Lp(epochs, delta_thr=delta_thr, lr=lr, lower_p=5, upper_p=50)
    ccl.f_reset()
    a3, b3 = ccl.cc_Linf_Lp(epochs, delta_thr=delta_thr, lr=lr, lower_p=10, upper_p=50)
    ccl.f_reset()
    a4, b4 = ccl.cc_Linf_softmax(epochs, delta_thr=delta_thr, lr=lr, lower_temp=2)
    ccl.f_reset()
    a5, b5 = ccl.cc_Linf_softmax(epochs, delta_thr=delta_thr, lr=lr, lower_temp=5)
    ccl.f_reset()
    a6, b6 = ccl.cc_Linf_softmax(epochs, delta_thr=delta_thr, lr=lr, lower_temp=10)
    ccl.f_reset()
    a7, b7 = ccl.cc_Linf_softmax(epochs, delta_thr=delta_thr, lr=lr, lower_temp=20)
    ccl.f_reset()
    a8, b8 = ccl.cc_Linf_softmax(epochs, delta_thr=delta_thr, lr=lr, lower_temp=40)
    ccl.f_reset_L2()
    a9, b9 = ccl.cc_Lp(epochs, lr=lr, p_norm=np.inf)

    # As = np.stack([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9])
    Bs = np.stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    results.append(Bs)
results = np.asarray(results)
# %%
# np.save('loss.npy', results)
results = np.load('loss.npy')
# %%
for result, file_name in zip(results, ['hp1', 'hp2', 'hp3', 'hp4']):
    plt.figure(figsize=(4, 4))
    plt.plot(result[0], label="$L^\infty$ loss")
    plt.plot(result[1], label='2~50')
    plt.plot(result[2], label='5~50')
    plt.plot(result[3], label='10~50')
    plt.ylim(0.1, 0.5)
    plt.xlabel('epochs')
    plt.ylabel('$L^\infty$ loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../exp_fig/loss_comp_Alg1_{file_name}.svg')
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.plot(result[0], label="$L^\infty$ loss")
    plt.plot(result[4], label='Alg2, 2')
    plt.plot(result[5], label='Alg2, 5')
    plt.plot(result[6], label='Alg2, 10')
    plt.plot(result[7], label='Alg2, 20')
    plt.plot(result[8], label='Alg2, 40')
    plt.ylim(0.1, 0.5)
    plt.xlabel('epochs')
    plt.ylabel('$L^\infty$ loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../exp_fig/loss_comp_Alg2_{file_name}.svg')
    plt.close()
# %%
