import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from config.params_universal_graph_embedding_model import B, N, E, D, H, C, K


class InputTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_input_layers=1, drop_prob=0.0):
        super(InputTransformer, self).__init__()

        self.input_dim = input_dim
        self.drop_prob = drop_prob
        self.hidden_dim = hidden_dim
        self.num_input_layers = num_input_layers
        self.output_dim = hidden_dim

        self.input_layers = nn.ModuleList()
        curr_input_dim = input_dim
        for i in range(self.num_input_layers):
            self.input_layers.append(nn.Linear(curr_input_dim, hidden_dim))
            curr_input_dim = hidden_dim

    def forward(self, x: (N, D), A: (N, N)):

        out = torch.spmm(A, x)
        out = out + x
        for i in range(self.num_input_layers):
            out = self.input_layers[i](out)
            out = F.selu(out)
        return out
