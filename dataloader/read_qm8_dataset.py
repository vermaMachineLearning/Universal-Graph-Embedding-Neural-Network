import networkx as nx
from scipy import sparse
import pandas as pd
import os
import scipy.io
import torch
import numpy as np
from sklearn import preprocessing
from utils.read_sdf_dataset import read_from_sdf
from rdkit import Chem
from operator import itemgetter


atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):

    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_list)\
              + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) \
              + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) \
              + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] \
              + one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) \
              + [atom.GetIsAromatic()]

    return np.array(results)


def read_qm8_data(folder, prefix):

    sdf_file = folder + '/' + prefix + '.sdf'
    csv_file = folder + '/' + prefix + '.csv'
    graph_list = read_from_sdf(sdf_file)
    num_graphs = len(graph_list)
    df = pd.read_csv(csv_file)
    Y = np.array(df)
    graph_labels = Y[:, 1:]
    assert (Y.shape[0] == num_graphs)
    num_targets = graph_labels.shape[-1]

    X = [G['node_labels'] for G in graph_list]
    unique_node_labels = list(set(x for l in X for x in l))
    num_unique_node_labels = len(unique_node_labels)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(unique_node_labels)
    onehot_encoder = preprocessing.OneHotEncoder(n_values=num_unique_node_labels, sparse=False)
    onehot_vec = np.array(range(0, num_unique_node_labels))
    onehot_encoder.fit(onehot_vec.reshape(-1, 1))

    process_graph_list = []
    for i in range(num_graphs):
        if (i + 1) % 1000 == 0:
            print('num graphs processed so far: ', i)

        curr_edge_list = graph_list[i]['edge_list']
        node_labels = graph_list[i]['node_labels']
        curr_num_nodes = len(node_labels)
        assert(curr_num_nodes == graph_list[i]['num_nodes'])
        assert(len(curr_edge_list) == graph_list[i]['num_edges'])

        node_labels = label_encoder.transform(node_labels)
        node_feature_matrix = onehot_encoder.transform(node_labels.reshape(-1, 1))

        G_nx = nx.Graph()
        G_nx.add_nodes_from(np.array(range(0, curr_num_nodes)))
        G_nx.add_edges_from(curr_edge_list)
        G_nx.remove_edges_from(G_nx.selfloop_edges())
        num_edges = G_nx.number_of_edges()
        nx_curr_num_nodes = G_nx.number_of_nodes()
        assert(curr_num_nodes == nx_curr_num_nodes)

        G = dict()
        edge_matrix = np.array(list(G_nx.edges()))
        row = torch.LongTensor(np.concatenate((edge_matrix[:, 0], edge_matrix[:, 1])))
        col = torch.LongTensor(np.concatenate((edge_matrix[:, 1], edge_matrix[:, 0])))
        row_col_idx = torch.zeros((2, 2 * num_edges), dtype=torch.int64)
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
        G['node_feature_matrix'] = torch.FloatTensor(node_feature_matrix)
        G['num_nodes'] = curr_num_nodes
        G['graph_label'] = graph_labels[i]
        G['node_degree_vec'] = np.array(node_degree)
        G['adj_matrix'] = nx.adjacency_matrix(G_nx, nodelist=range(0, G['num_nodes']))
        G['node_labels'] = np.array(node_labels)

        assert (G['node_degree_vec'].shape[0] == G['num_nodes'])
        assert (G['adj_matrix'].shape[0] == G['num_nodes'])
        assert (G['node_feature_matrix'].shape[0] == G['num_nodes'])
        assert (G['node_feature_matrix'].shape[1] == num_unique_node_labels)
        assert (G_nx.number_of_selfloops() == 0)

        process_graph_list.append(G)

    return process_graph_list, num_targets
