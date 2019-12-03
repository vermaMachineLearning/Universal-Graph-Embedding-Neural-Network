from scipy import sparse
import numpy as np
from utils.fast_wl_kernel import wl_kernel, wl_kernel_batch
import time
from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA, NMF
from sklearn import random_projection
import logging


def compute_reduce_wl_kernel(graph_list, num_iter=5, reduce_feature_dim=1000):

    logging.info('Preparing WL kernel input data...')
    t = time.time()
    node_labels = graph_list[0]['node_degree_vec']
    graph_indicator = 0 * np.ones(graph_list[0]['num_nodes'], dtype=np.int64)
    for i in range(1, len(graph_list)):
        if (i + 1) % 1000 == 0:
            logging.info('num graphs processed so far: ' + str(i + 1))
        node_labels = np.concatenate((node_labels, graph_list[i]['node_degree_vec']))
        graph_indicator = np.concatenate((graph_indicator, i * np.ones(graph_list[i]['num_nodes'], dtype=np.int64)))
    logging.info('Time Taken: ' + str(time.time() - t))

    logging.info('Computing WL Kernel...')
    t = time.time()
    feature_matrix = wl_kernel_batch([graph['adj_matrix'] for graph in graph_list], node_labels, graph_indicator, num_iter, compute_kernel_matrix=False)  # TODO: quality testing
    logging.info('Feature matrix shape: ' + str(feature_matrix.shape))
    logging.info('Time Taken: ' + str(time.time() - t))

    feature_matrix = sparse.csr_matrix(feature_matrix)
    logging.info('Performing dimension reduction...')
    t = time.time()
    transformer = random_projection.SparseRandomProjection(n_components=reduce_feature_dim)
    feature_matrix_reduce = transformer.fit_transform(feature_matrix)
    feature_matrix_reduce[feature_matrix_reduce < 0] = 0
    feature_matrix_reduce[feature_matrix_reduce > 0] = 1
    logging.info('feature matrix density: ' + str(feature_matrix_reduce.getnnz() / (feature_matrix_reduce.shape[0] * feature_matrix_reduce.shape[1])))
    feature_matrix_reduce = feature_matrix_reduce.todense()
    logging.info('Time Taken: ' + str(time.time() - t))

    return feature_matrix_reduce


def compute_full_wl_kernel(graph_list, num_iter=5, type_node_labels='degree'):

    logging.info('Preparing WL kernel input data...')
    t = time.time()
    if type_node_labels == 'degree':
        node_labels = graph_list[0]['node_degree_vec']
    elif type_node_labels == 'node_label':
        node_labels = graph_list[0]['node_labels']

    graph_indicator = 0 * np.ones(graph_list[0]['num_nodes'], dtype=np.int64)
    for i in range(1, len(graph_list)):
        if (i + 1) % 1000 == 0:
            logging.info('num graphs processed so far: ' + str(i + 1))
        if type_node_labels == 'degree':
            curr_node_labels = graph_list[i]['node_degree_vec']
        elif type_node_labels == 'node_label':
            curr_node_labels = graph_list[i]['node_labels']
        node_labels = np.concatenate((node_labels, curr_node_labels))
        graph_indicator = np.concatenate((graph_indicator, i * np.ones(graph_list[i]['num_nodes'], dtype=np.int64)))
    logging.info('Time Taken: ' + str(time.time() - t))

    logging.info('Computing WL Kernel...')
    t = time.time()
    feature_matrix = wl_kernel_batch([graph['adj_matrix'] for graph in graph_list], node_labels, graph_indicator, num_iter, compute_kernel_matrix=False, normalize_feature_matrix=True)
    logging.info('Feature matrix shape: ' + str(feature_matrix.shape))
    logging.info('Time Taken: ' + str(time.time() - t))

    return feature_matrix
