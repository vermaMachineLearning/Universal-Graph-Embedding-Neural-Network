from scipy import sparse
import numpy as np
from utils.fast_fgsd_features import fgsd_features
import time
from sklearn import random_projection


def compute_reduce_fgsd_features(graph_list):

    print('Computing fgsd features...')
    t = time.time()
    feature_matrix = fgsd_features([graph['adj_matrix'] for graph in graph_list])  # TODO: quality testing
    print('Feature matrix shape: ', feature_matrix.shape)
    print('Time Taken: ', time.time() - t)

    print('Performing dimension reduction...')
    t = time.time()
    transformer = random_projection.SparseRandomProjection(n_components=1000)
    feature_matrix_reduce = transformer.fit_transform(feature_matrix)
    feature_matrix_reduce[feature_matrix_reduce < 0] = 0
    feature_matrix_reduce[feature_matrix_reduce > 0] = 1
    print('feature matrix density: ', feature_matrix_reduce.getnnz() / (feature_matrix_reduce.shape[0] * feature_matrix_reduce.shape[1]))
    feature_matrix_reduce = feature_matrix_reduce.todense()
    print('Time Taken: ', time.time() - t)

    return feature_matrix_reduce
