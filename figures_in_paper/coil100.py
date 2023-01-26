# %%
import os
import sys

os.chdir('experiments')
sys.path.append('../')

import joblib
import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image
from persim import plot_diagrams
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from umap import UMAP

from circularcoordinates import weighted_circular_coordinate, CircCoordLn

# %%
path_dir = '../COIL'
file_list = os.listdir(path_dir)
file_list.sort()
images = [np.asarray(Image.open(f'{path_dir}/{el}')) for el in tqdm(file_list)]
images = np.asarray(images)
images_flatten = images.reshape(len(images), -1)
images_flatten_torch = torch.from_numpy(images_flatten) / 256
obj_list = np.asarray([int(el.split('__')[0][3:]) - 1 for el in file_list])
label = np.asarray([int(el.split('__')[0][3:]) for el in file_list]) - 1
prime = 47
# %%
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
umap = UMAP(random_state=42)
# %%
pca_result = pca.fit_transform(images_flatten)
tsne_result = tsne.fit_transform(images_flatten)
umap_result = umap.fit_transform(images_flatten)
# %%
plt.figure(figsize=(6, 6))
plt.scatter(*pca_result.T, s=1)
plt.axis('off')
plt.tight_layout()
# plt.savefig('../coil_figures/pca_result.svg')
plt.savefig('../coil_figures/pca_result.png', dpi=300)
plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(*tsne_result.T, s=1)
plt.axis('off')
plt.tight_layout()
plt.savefig('../coil_figures/tsne_result.png', dpi=300)
plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(*tsne_result.T, s=1, c=['C1' if el else 'C0' for el in (label == 13)], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../coil_figures/tsne_label_ex0.png', dpi=300)
plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(*tsne_result.T, s=1, c=['C1' if el else 'C0' for el in (label == 76)], cmap='hsv')
plt.axis('off')
plt.tight_layout()
plt.savefig('../coil_figures/tsne_label_ex1.png', dpi=300)
plt.close()
# %%
if os.path.exists('coil_ripser.pkl'):
    ripser_result = joblib.load('coil_ripser.pkl')
else:
    dist_mat = torch.cdist(images_flatten_torch, images_flatten_torch)
    print(dist_mat.shape)
    ripser_result = ripser(dist_mat.numpy(), distance_matrix=True, coeff=47, do_cocycles=True)
    joblib.dump(ripser_result, 'coil_ripser.pkl', )
# %%
plt.figure(figsize=(4, 4))
plot_diagrams(ripser_result['dgms'])
plt.tight_layout()
plt.savefig('../coil_figures/persistence_diagram.png', dpi=300)
# plt.savefig('../coil_figures/persistence_diagram.svg')
plt.close()
# %%
if os.path.exists('coil_circ_results.pkl'):
    circ_results = joblib.load('coil_circ_results.pkl')
else:
    circ_results = [
        weighted_circular_coordinate(ripser_result, ripser_result=True, prime=prime, cocycle_n=i, weight_ft=None)
        for i in tqdm(range(100))
    ]
    joblib.dump(circ_results, 'coil_circ_results.pkl')
print(len(circ_results))
# %%
ccl = CircCoordLn(np.random.randn(5, 2))
ccl.ripser_result = ripser_result
ccl.dgm = ripser_result['dgms'][1]
ccl.argsort = np.argsort(ccl.dgm[:, 1] - ccl.dgm[:, 0])[::-1]
ccl.f_value = torch.zeros(len(file_list))
# %%
hyper_params = [
    {'epoch': 100, 'lr': 0.001, 'p_norm': 2},
    {'epoch': 5000, 'lr': 0.05, 'p_norm': 4},
    {'epoch': 5000, 'lr': 0.05, 'p_norm': 6},
    {'epoch': 5000, 'lr': 0.05, 'p_norm': 10},
    {'epoch': 10000, 'lr': 0.01, 'p_norm': 20},
    {'epoch': 10000, 'lr': 0.01, 'p_norm': 30},
]
lp_results = []
for i in tqdm(range(100)):
    ccl.setup(i)
    orig_1 = ccl.cc_original()
    ccl.f_reset_L2()
    for param in hyper_params:
        ccl.cc_Lp(**param, delta_thr=1e-7, max_count=500)
    lp_results.append(
        np.mod(ccl.cc_Linf_Lp(10000, 0.0001, 0.001, 30, 30)[0], 1.0)
    )
lp_results = np.asarray(lp_results)
# %%
# np.save('coil_Linfty.npy', lp_results)
lp_results = np.load('coil_Linfty.npy')
# %%
indices = [
    [76, 21],
    [12, 9],
    [56, 1],
    [64, 15],
    [39, 2],
    [74, 26],
    [59, 28],
    [30, 30],
    [97, 57],
    [89, 97]
]
# %%
for i in range(10):
    print(i)
    i_indices = [el for el in label == indices[i][0]]
    i_color = ['C1' if el else 'C0' for el in label == indices[i][0]]
    i_size = [30 if el else 1 for el in label == indices[i][0]]

    k = 0
    for dim_emb in [tsne_result, pca_result]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(*dim_emb[i_indices].T)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.close()
        k += 1
        print(k)
        j = 0
        for result in [circ_results, lp_results]:
            j += 1
            if (i, j, k) not in [
                (0, 1, 1),
                (0, 1, 2),
                (5, 1, 1),
                (5, 2, 1),
            ]:
                continue
            print(j)
            plt.figure(figsize=(6, 6))
            plt.scatter(*dim_emb.T, s=1, c=result[indices[i][1]], cmap='hsv')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'../coil_figures/{i}_{j}_{k}_result.png', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.scatter(*dim_emb.T, s=1, c=i_color, cmap='hsv')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'../coil_figures/{i}_{j}_{k}_label.png', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.scatter(*dim_emb.T, s=i_size, c=result[indices[i][1]], cmap='hsv', alpha=0.5)
            plt.axis('off')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.tight_layout()
            plt.savefig(f'../coil_figures/{i}_{j}_{k}_zoom.png', dpi=300)
            plt.close()

# %%
