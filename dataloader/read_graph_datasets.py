import glob
import pandas as pd
import torch
import numpy as np
from dataloader.data import Data
from sklearn import preprocessing
import networkx as nx
import os
import scipy.io
from scipy import sparse


names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]


def read_graph_data(folder, prefix):

    edge_list_file = folder + '/' + prefix + '_A.txt'
    graph_ind_file = folder + '/' + prefix + '_graph_indicator.txt'
    graph_labels_file = folder + '/' + prefix + '_graph_labels.txt'
    node_label_file = folder + '/' + prefix + '_node_labels.txt'
    fgsd_mat_file = os.path.dirname(folder) + '/' + prefix + '_X_fgsd.mat'

    df_edge_list = pd.read_csv(edge_list_file, header=None)
    df_graph_ind = pd.read_csv(graph_ind_file, header=None)
    df_graph_labels = pd.read_csv(graph_labels_file, header=None)

    edge_index_array = np.array(df_edge_list, dtype=np.int64)
    graph_labels = np.array(df_graph_labels, dtype=np.int64).reshape(-1)

    graph_ind = np.array(df_graph_ind, dtype=np.int64).reshape(-1)
    graph_ind_encoder = preprocessing.LabelEncoder()
    graph_ind = graph_ind_encoder.fit_transform(graph_ind)
    graph_ind_vals, graph_ind_start_idx = np.unique(graph_ind, return_index=True)
    num_graphs = len(graph_ind_vals)
    big_num_edges = edge_index_array.shape[0]

    if os.path.exists(node_label_file):
        df_node_labels = pd.read_csv(node_label_file, header=None)
        node_labels = np.array(df_node_labels, dtype=np.int64).reshape(-1)
        node_label_encoder = preprocessing.LabelEncoder()
        node_labels = node_label_encoder.fit_transform(node_labels)
    else:
        node_labels = np.zeros(len(graph_ind), dtype=np.int64)

    graph_label_encoder = preprocessing.LabelEncoder()
    graph_labels = graph_label_encoder.fit_transform(graph_labels)
    num_classes = graph_labels.max() + 1

    load_fgsd_features = False
    if os.path.exists(fgsd_mat_file):
        mat = scipy.io.loadmat(fgsd_mat_file)
        fgsd_feature_matrix = sparse.csr_matrix(mat['X'], dtype=np.float64)
        preprocessing.normalize(fgsd_feature_matrix, norm='l2', axis=1, copy=False, return_norm=False)
        assert (fgsd_feature_matrix.shape[0] == num_graphs)
        load_fgsd_features = True

    assert (graph_ind[-1] + 1 == num_graphs)
    assert (len(graph_labels) == num_graphs)
    assert (len(graph_ind) == len(node_labels))
    assert (node_labels.min() == 0)
    assert (node_labels.max() + 1 == len(np.unique(node_labels)))

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    big_node_feature_matrix = onehot_encoder.fit_transform(node_labels.reshape(-1, 1))
    big_node_feature_matrix = torch.FloatTensor(big_node_feature_matrix)

    graph_ind_start_idx = np.concatenate((graph_ind_start_idx, [len(graph_ind)]))
    graph_list = []
    prev_node_start_id = 1
    prev_edge_end_idx = 0
    for i in range(num_graphs):
        if (i+1) % 1000 == 0:
            print('num graphs processed so far: ', i)

        curr_num_nodes = graph_ind_start_idx[i+1] - graph_ind_start_idx[i]
        node_feature_matrix = big_node_feature_matrix[graph_ind_start_idx[i]: graph_ind_start_idx[i+1]]
        graph_node_labels = node_labels[graph_ind_start_idx[i]: graph_ind_start_idx[i+1]]

        if i == num_graphs-1:
            curr_edge_end_idx = big_num_edges
        else:
            curr_edge_end_idx = np.argmax(edge_index_array[:, 0] > graph_ind_start_idx[i + 1])
        curr_edge_list = edge_index_array[prev_edge_end_idx: curr_edge_end_idx]
        if curr_edge_list.size == 0:
            prev_node_start_id = prev_node_start_id + curr_num_nodes
            continue
        curr_edge_list = curr_edge_list - prev_node_start_id

        assert(curr_edge_list.min() >= 0)
        assert(curr_edge_list.max() <= curr_num_nodes - 1)

        prev_edge_end_idx = curr_edge_end_idx
        prev_node_start_id = graph_ind_start_idx[i+1] + 1

        G_nx = nx.Graph()
        G_nx.add_edges_from(curr_edge_list)
        G_nx.remove_edges_from(G_nx.selfloop_edges())
        num_edges = G_nx.number_of_edges()

        G = dict()
        edge_matrix = np.array(list(G_nx.edges()))
        row = torch.LongTensor(np.concatenate((edge_matrix[:, 0], edge_matrix[:, 1])))
        col = torch.LongTensor(np.concatenate((edge_matrix[:, 1], edge_matrix[:, 0])))
        row_col_idx = torch.zeros((2, 2*num_edges), dtype=torch.int64)
        row_col_idx[0] = row
        row_col_idx[1] = col

        node_degree = []
        for node_id in range(curr_num_nodes):
            if G_nx.has_node(node_id):
                node_degree.append(G_nx.degree[node_id])
            else:
                node_degree.append(0)

        G['graph_id'] = i
        G['edge_matrix'] = row_col_idx
        G['node_feature_matrix'] = node_feature_matrix
        G['node_labels'] = graph_node_labels
        G['num_nodes'] = curr_num_nodes
        G['graph_label'] = graph_labels[i]
        G['node_degree_vec'] = np.array(node_degree)
        G['adj_matrix'] = nx.adjacency_matrix(G_nx, nodelist=range(0, G['num_nodes']))
        if load_fgsd_features:
            G['fgsd_features'] = fgsd_feature_matrix[i]

        assert(G['node_degree_vec'].shape[0] == G['num_nodes'])
        assert(G['adj_matrix'].shape[0] == G['num_nodes'])
        assert(G['node_feature_matrix'].shape[0] == G['num_nodes'])
        assert(G['node_feature_matrix'].shape[1] == node_labels.max() + 1)
        assert(G_nx.number_of_selfloops() == 0)

        graph_list.append(G)

    return graph_list, num_classes




