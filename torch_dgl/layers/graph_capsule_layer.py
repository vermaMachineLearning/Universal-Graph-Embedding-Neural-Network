import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCapsuleConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gfc_layers=2, num_stats_in=1, num_stats_out=1):
        super(GINConv, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stats_in = num_stats_in
        self.num_stats_out = num_stats_out
        self.num_gfc_layers = num_gfc_layers

        self.stat_layers = nn.ModuleList()
        for _ in range(self.num_stats_out):
            gfc_layers = nn.ModuleList()
            curr_input_dim = input_dim * num_stats_in
            for _ in range(self.num_gfc_layers):
                gfc_layers.append(nn.Linear(curr_input_dim, hidden_dim))
                curr_input_dim = hidden_dim
            self.stat_layers.append(gfc_layers)

    def forward(self, x_in, A):

        x = x_in
        output = []
        for i in range(self.num_stats_out):
            out = torch.spmm(A, x) + x
            for j in range(self.num_gfc_layers):
                out = self.stat_layers[i][j](out)
                out = F.selu(out)
            output.append(out)
            x = torch.mul(x, x_in)

        output = torch.cat(output, 1)
        return output
