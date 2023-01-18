import numpy as np
from ripser import ripser
from scipy import sparse


def weighted_circular_coordinate(data, distance_matrix=False, ripser_result=False, prime=47, cocycle_n=None, eps=None,
                                 weight_ft: callable = None, return_aux=False):
    if not ripser_result:
        ripser_result = ripser(data, distance_matrix=distance_matrix, coeff=prime, do_cocycles=True)
    else:
        ripser_result = data

    dist_mat = ripser_result['dperm2all']
    n_vert = len(dist_mat)

    argsort_eps = np.argsort(np.diff(ripser_result['dgms'][1], 1)[:, 0])[::-1]
    if cocycle_n is None:
        cocycle_n = argsort_eps[0]
    else:
        cocycle_n = argsort_eps[cocycle_n]

    if eps is None:
        birth, death = ripser_result['dgms'][1][cocycle_n]
        eps = (birth + death) / 2

    # Delta
    edges = np.asarray((dist_mat <= eps).nonzero()).T
    n_edges = len(edges)
    I = np.c_[np.arange(n_edges), np.arange(n_edges)]
    I = I.flatten()
    J = edges.flatten()
    V = np.c_[-1 * np.ones(n_edges), np.ones(n_edges)]
    V = V.flatten()
    delta = sparse.coo_matrix((V, (I, J)), shape=(n_edges, n_vert))

    # Cocycle
    cocycle = ripser_result["cocycles"][1][cocycle_n]
    val = cocycle[:, 2]
    val[val > (prime - 1) / 2] -= prime
    Y = sparse.coo_matrix((val, (cocycle[:, 0], cocycle[:, 1])), shape=(n_vert, n_vert))
    Y = Y - Y.T
    cocycle = np.asarray(Y[edges[:, 0], edges[:, 1]])[0]

    # Minimize
    if weight_ft is None:
        mini = sparse.linalg.lsqr(delta, cocycle)[0]
    else:
        new_delta, new_cocycle = weight_ft(delta, cocycle, dist_mat, edges)
        mini = sparse.linalg.lsqr(new_delta, new_cocycle)[0]

    if return_aux:
        return delta, mini, cocycle
    else:
        return np.mod(mini, 1.0)


def weight_ft_0(k, t=None):
    def _weight_ft(delta, cocycle, dist_mat, edges):
        nonlocal t
        if t is None:
            tmp = dist_mat[edges[:, 0], edges[:, 1]]
            t = 0.1 * np.mean(tmp[tmp != 0])
            print(t)
        G = np.exp(-(dist_mat ** 2) / (4 * t))
        G = G / ((4 * np.pi * t) ** (k / 2))
        P = np.mean(G, axis=0)
        P_inv = np.diag(1 / P)
        W = G @ P_inv
        D = np.sum(W, axis=1)
        L_w = P_inv @ ((D @ P - G) / (t * len(dist_mat))) @ P_inv
        metric_weight = L_w[edges[:, 0], edges[:, 1]]
        metric_weight = np.maximum(metric_weight, 0)  # for safety
        sqrt_weight = np.sqrt(metric_weight)

        new_delta = delta.multiply(sqrt_weight[:, None])
        new_cocycle = sqrt_weight * cocycle

        return new_delta, new_cocycle

    return _weight_ft


def weight_ft_with_degree_meta(ft: callable):
    # ft(degree1, degree2) -> c
    def _weight_ft(delta, cocycle, dist_mat, edges):
        ndarray_delta = delta.toarray()
        degrees = np.abs(ndarray_delta).sum(0) / 2
        weight = []
        for el in ndarray_delta:
            tmp = np.where(el != 0)[0]
            if len(tmp) == 0:
                weight.append(1)
            else:
                tmp = np.sort(tmp)
                weight.append(
                    ft(degrees[tmp[0]], degrees[tmp[1]])
                )

        weight = np.asarray(weight)
        new_delta = delta.multiply(weight[:, None])
        new_cocycle = weight * cocycle

        return new_delta, new_cocycle

    return _weight_ft
