import torch.utils.data
import networkx as nx
import time
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix, coo_matrix


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, dataset_name=None, batch_size=1, shuffle=True, batch_prep_func=None, **kwargs):

        def batch_collate(batch_graph_list):
            t = time.time()
            batch = dict()
            num_graphs = len(batch_graph_list)

            batch_sample_idx = []
            batch_node_feature_matrix = []
            batch_graph_labels = []
            batch_edge_matrix = []
            # batch_adj_mask_matrix = []
            prev_num_nodes = 0

            for i, G in enumerate(batch_graph_list):
                num_nodes = G['num_nodes']
                edge_matrix = G['edge_matrix']
                label = G['graph_label']
                node_feature_matrix = G['node_feature_matrix']
                # node_feature_matrix = torch.ones(num_nodes, 1)

                curr_num_nodes = prev_num_nodes
                edge_matrix = edge_matrix + int(curr_num_nodes)
                prev_num_nodes = curr_num_nodes + num_nodes

                # mask_matrix = np.ones((num_nodes, num_nodes))
                # np.fill_diagonal(mask_matrix, 0)
                # adj_mask_matrix = torch.LongTensor(coo_matrix(mask_matrix).nonzero()) + int(curr_num_nodes)
                # batch_adj_mask_matrix.append(adj_mask_matrix)

                batch_edge_matrix.append(edge_matrix)
                batch_sample_idx.append(i * torch.ones(num_nodes, dtype=torch.int64))
                batch_node_feature_matrix.append(node_feature_matrix)
                batch_graph_labels.append(label)

            total_num_nodes = prev_num_nodes

            edge_matrix = torch.cat(batch_edge_matrix, dim=1)
            total_num_edges = edge_matrix.shape[-1]
            val = torch.ones(total_num_edges)
            A = torch.sparse.FloatTensor(edge_matrix, val, torch.Size([total_num_nodes, total_num_nodes]))

            # adj_mask_matrix = torch.cat(batch_adj_mask_matrix, dim=1)
            # val = torch.ones(adj_mask_matrix.shape[-1])
            # A_mask = torch.sparse.FloatTensor(adj_mask_matrix, val, torch.Size([total_num_nodes, total_num_nodes]))

            batch_sample_idx = torch.cat(batch_sample_idx)
            sparse_idx = torch.stack((batch_sample_idx, torch.arange(0, int(total_num_nodes), dtype=torch.long)))
            val = torch.ones(total_num_nodes)
            batch_sample_matrix = torch.sparse.FloatTensor(sparse_idx, val, torch.Size([num_graphs, total_num_nodes]))

            batch['edge_matrix'] = edge_matrix
            batch['adjacency_matrix'] = A
            batch['node_feature_matrix'] = torch.cat(batch_node_feature_matrix)
            batch['graph_labels'] = torch.LongTensor(batch_graph_labels)
            batch['batch_sample_matrix'] = batch_sample_matrix
            batch['num_graphs'] = num_graphs
            batch['adjacency_mask'] = None

            if self.batch_prep_func is not None:
                batch = self.batch_prep_func(batch, batch_graph_list, dataset_name=dataset_name)

            batch['prep_time'] = time.time() - t
            return batch

        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=batch_collate, **kwargs)
        self.batch_prep_func = batch_prep_func


