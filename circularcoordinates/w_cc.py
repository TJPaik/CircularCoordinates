import numpy as np
from ripser import ripser

from .circular_coordinates import circular_coordinate


def weighted_circular_coordinate(data, ft: callable, prime: int = 47, order: int = 0, eps=None):
    ripser_result = ripser(data, coeff=prime, do_cocycles=True)
    arg_eps = np.argsort(np.diff(ripser_result['dgms'][1], axis=1).flatten())[::-1][order]
    circ = circular_coordinate(prime=prime)
    if eps is None:
        eps = np.mean(ripser_result['dgms'][1][arg_eps])
    _, vertex_values, _ = circ.circular_coordinate(ripser_result, prime, arg_eps=arg_eps, weight=ft, eps=eps)
    return vertex_values


def weight_degree_sum(delta, cocycle, distances):
    nparray_abs = np.abs(delta).toarray()
    degrees = nparray_abs.sum(0) / 2
    result = []
    for el in nparray_abs:
        tmp = np.where(el != 0)[0]
        if len(tmp) == 0:
            result.append(1)
        else:
            result.append(np.sum(degrees[tmp]))
    return np.asarray(result)


def weight_degree_sum_meta(ft: callable):
    if ft is None:
        return None

    def tmp(delta, cocycle, distances):
        return ft(weight_degree_sum(delta, cocycle, distances))

    return tmp
