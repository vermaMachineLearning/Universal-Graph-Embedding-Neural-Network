from scipy import sparse
import numpy as np
import time
from fast_histogram import histogram1d
import matlab.engine

eng = matlab.engine.start_matlab()


def fgsd_features(graph_list):

    S_max = 0
    S_list = []
    print('Computing pseudo inverse...')
    t = time.time()
    for i, A in enumerate(graph_list):
        if (i + 1) % 1000 == 0:
            print('num graphs processed so far: ', i + 1)
        A = np.array(A.todense(), dtype=np.float32)
        D = np.sum(A, axis=0)
        L = np.diag(D) - A

        ones_vector = np.ones(L.shape[0])
        try:
            fL = np.linalg.pinv(L)
        except np.linalg.LinAlgError:
            fL = np.array(eng.fgsd_fast_pseudo_inverse(matlab.double(L.tolist()), nargout=1))
        fL[np.isinf(fL)] = 0
        fL[np.isnan(fL)] = 0

        S = np.outer(np.diag(fL), ones_vector) + np.outer(ones_vector, np.diag(fL)) - 2 * fL
        if S.max() > S_max:
            S_max = S.max()
        S_list.append(S)

    print('S_max: ', S_max)
    print('Time Taken: ', time.time() - t)

    feature_matrix = []
    nbins = 1000000
    range_hist = (0, S_max)
    print('Computing histogram...')
    t = time.time()
    for i, S in enumerate(S_list):
        if (i + 1) % 1000 == 0:
            print('num graphs processed so far: ', i + 1)
        # hist, _ = np.histogram(S.flatten(), bins=nbins, range=range_hist)
        hist = histogram1d(S.flatten(), bins=nbins, range=range_hist)
        hist = sparse.csr_matrix(hist)
        feature_matrix.append(hist)
    print('Time Taken: ', time.time() - t)

    feature_matrix = sparse.vstack(feature_matrix)
    return feature_matrix
