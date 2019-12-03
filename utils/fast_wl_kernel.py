import numpy as np
import sympy
from scipy import sparse
from itertools import product
import networkx as nx
import scipy.io
from sklearn.preprocessing import normalize


prime_numbers_list = np.load('data/prime_numbers_list_v2.npy')


def uniquetol(ar, tol=1e-12, return_index=False, return_inverse=False, return_counts=False, axis=None):
    ar = np.asanyarray(ar)
    if axis is None:
        return unique1dtol(ar, tol, return_index, return_inverse, return_counts)
    if not (-ar.ndim <= axis < ar.ndim):
        raise ValueError('Invalid axis kwarg specified for unique')

    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = unique1dtol(consolidated, tol, return_index, return_inverse, return_counts)
    if not (return_index or return_inverse or return_counts):
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return (uniq,) + output[1:]


def unique1dtol(ar, tol, return_index=False, return_inverse=False, return_counts=False):

    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.intp),)
            if return_inverse:
                ret += (np.empty(0, np.intp),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], np.absolute(aux[1:] - aux[:-1]) >= tol * np.max(np.absolute(aux[:]))))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def wl_transformation(A, node_labels):

    num_labels = max(node_labels) + 1
    log_primes = np.log(prime_numbers_list[0:num_labels])

    signatures = node_labels + A.dot(log_primes[node_labels])
    _, new_labels = uniquetol(signatures.flatten('F'), return_inverse=True)
    return new_labels


def wl_kernel(A, node_labels, graph_ind, num_iterations):

    num_graphs = max(graph_ind) + 1
    K = sparse.csr_matrix((num_graphs, num_graphs))
    feature_matrix = None

    for i in range(num_iterations+1):

        num_nodes = len(graph_ind)
        num_node_labels = max(node_labels) + 1
        counts = sparse.coo_matrix((np.ones(num_nodes), (graph_ind, node_labels)), shape=(num_graphs, num_node_labels))
        if feature_matrix is None:
            feature_matrix = counts
        else:
            feature_matrix = sparse.hstack([feature_matrix, counts])

        # K_new = counts.dot(counts.transpose())
        # K = K + K_new
        node_labels = wl_transformation(A, node_labels)

    return K, feature_matrix


def wl_transformation_batch(A_batch, node_labels):

    num_labels = max(node_labels) + 1
    log_primes = np.log(prime_numbers_list[0:num_labels])

    signatures = []
    prev_total_num_nodes = 0
    for A in A_batch:
        curr_total_num_nodes = prev_total_num_nodes + A.shape[0]
        curr_node_labels = node_labels[prev_total_num_nodes: curr_total_num_nodes]
        out = curr_node_labels + A.dot(log_primes[curr_node_labels])
        signatures.append(out)
        prev_total_num_nodes = curr_total_num_nodes

    signatures = np.concatenate(signatures, axis=0)
    _, new_labels = uniquetol(signatures.flatten('F'), return_inverse=True)
    return new_labels


def wl_kernel_batch(A_batch, node_labels, graph_ind, num_iterations, compute_kernel_matrix=False, normalize_feature_matrix=False):

    num_graphs = max(graph_ind) + 1
    feature_matrix = None

    for i in range(num_iterations+1):
        num_nodes = len(graph_ind)
        num_node_labels = max(node_labels) + 1
        counts = sparse.coo_matrix((np.ones(num_nodes), (graph_ind, node_labels)), shape=(num_graphs, num_node_labels))
        if feature_matrix is None:
            feature_matrix = counts
        else:
            feature_matrix = sparse.hstack([feature_matrix, counts])
        node_labels = wl_transformation_batch(A_batch, node_labels)

    feature_matrix = sparse.csr_matrix(feature_matrix)
    if normalize_feature_matrix:
        normalize(feature_matrix, norm='l2', axis=1, copy=False, return_norm=False)
    if compute_kernel_matrix:
        K = feature_matrix.dot(feature_matrix.transpose())
        return K, feature_matrix
    else:
        return feature_matrix
