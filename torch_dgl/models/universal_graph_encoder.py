import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_dgl.layers.graph_capsule_layer import GraphCapsuleConv
import torch.nn.functional as F
from config.params_universal_graph_embedding_model import B, N, E, D, H, C


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gconv_layers, num_gfc_layers, drop_prob):
        super(GraphEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.num_gconv_layers = num_gconv_layers
        self.num_gfc_layers = num_gfc_layers

        self.gconv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        curr_input_dim = input_dim
        for i in range(self.num_gconv_layers):
            self.gconv_layers.append(GraphCapsuleConv(curr_input_dim, hidden_dim, num_gfc_layers=num_gfc_layers))
            self.batchnorm_layers.append(torch.nn.BatchNorm1d(hidden_dim))
            curr_input_dim = curr_input_dim + hidden_dim

    def forward(self, x: (N, D), edge_index: (2, E)):

        x_prev = x
        for i in range(self.num_gconv_layers):
            x_curr: (N, H) = F.selu(self.gconv_layers[i](x_prev, edge_index))
            x_curr: (N, H) = self.batchnorm_layers[i](x_curr)
            x_prev = torch.cat((x_prev, x_curr), dim=-1)
        return x_prev
