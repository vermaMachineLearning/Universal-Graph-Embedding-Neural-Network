import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_dgl.models.universal_graph_encoder import GraphEncoder
from config.params_universal_graph_embedding_model import B, N, E, D, H, C, K
from torch_dgl.layers.ugd_input_layer import InputTransformer


class UniversalGraphEmbedder(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_gconv_layers, num_gfc_layers, drop_prob, use_fgsd_features=False, use_spk_features=False):
        super(UniversalGraphEmbedder, self).__init__()

        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.num_encoder_layers = num_gconv_layers
        self.hidden_dim = hidden_dim
        self.num_gfc_layers = num_gfc_layers
        self.use_fgsd_features = use_fgsd_features
        self.use_spk_features = use_spk_features
        self.drop_prob_kernel = 0

        self.input_transform = InputTransformer(input_dim, hidden_dim, num_input_layers=1)
        self.graph_encoder = GraphEncoder(hidden_dim, hidden_dim, num_gconv_layers, num_gfc_layers, drop_prob)
        self.output_dim = hidden_dim + num_gconv_layers * hidden_dim

        self.num_fc_layers = 1
        self.fc_layers = nn.ModuleList()
        curr_input_dim = self.output_dim
        for i in range(self.num_fc_layers):
            self.fc_layers.append(Linear(curr_input_dim, hidden_dim))
            curr_input_dim = hidden_dim

        self.class_fc_layer = Linear(hidden_dim, num_classes)

        self.adj_linear_layer = Linear(self.output_dim, self.output_dim)

        self.kernel_linear_layer = Linear(self.output_dim, self.output_dim)
        if use_fgsd_features:
            self.fgsd_linear_layer = Linear(self.output_dim, self.output_dim)

        if use_spk_features:
            self.spk_linear_layer = Linear(self.output_dim, self.output_dim)

    def forward(self, x: (N, D), A: (N, N), A_mask: (N, N), batch_sample_matrix: (B, N)):

        x: (N, H) = self.input_transform(x, A)
        x: (N, H) = self.graph_encoder(x, A)
        x_emb: (B, H) = torch.spmm(batch_sample_matrix, x)

        # A_pred: (N, N) = torch.matmul(self.adj_linear_layer(x), x.t())
        # A_pred: (N, N) = torch.matmul(x, x.t())
        # A_pred: (N, N) = torch.mul(A_pred, A_mask.to_dense())
        A_pred = None

        x_curr = x_emb
        for i in range(self.num_fc_layers):
            x_curr: (B, H) = F.selu(self.fc_layers[i](x_curr))
            x_curr: (B, H) = F.dropout(x_curr, p=self.drop_prob, training=self.training)

        x: (B, C) = self.class_fc_layer(x_curr)
        class_logits: (B, C) = F.log_softmax(x, dim=-1)

        # kernel_matrix_pred = torch.matmul(self.kernel_linear_layer(x_emb), x_emb.t())
        # kernel_matrix_pred = F.dropout(kernel_matrix_pred, p=self.drop_prob_kernel, training=self.training)
        # kernel_matrix_pred = torch.sigmoid(kernel_matrix_pred)

        kernel_matrix_pred = torch.matmul(x_emb, x_emb.t())
        kernel_matrix_pred = F.dropout(kernel_matrix_pred, p=self.drop_prob_kernel, training=self.training)
        kernel_matrix_pred = torch.sigmoid(kernel_matrix_pred)

        # x: (B, H) = F.normalize(x_emb, p=2, dim=1)
        # kernel_matrix_pred: (B, B) = torch.matmul(x, x.t())

        fgsd_kernel_matrix_pred = None
        if self.use_fgsd_features:
            fgsd_kernel_matrix_pred = torch.matmul(self.fgsd_linear_layer(x_emb), x_emb.t())
            fgsd_kernel_matrix_pred = F.dropout(fgsd_kernel_matrix_pred, p=self.drop_prob_kernel, training=self.training)
            fgsd_kernel_matrix_pred = torch.sigmoid(fgsd_kernel_matrix_pred)

        spk_kernel_matrix_pred = None
        if self.use_spk_features:
            spk_kernel_matrix_pred = torch.matmul(self.spk_linear_layer(x_emb), x_emb.t())
            spk_kernel_matrix_pred = F.dropout(spk_kernel_matrix_pred, p=self.drop_prob_kernel, training=self.training)
            spk_kernel_matrix_pred = torch.sigmoid(spk_kernel_matrix_pred)

        return class_logits, kernel_matrix_pred, fgsd_kernel_matrix_pred, spk_kernel_matrix_pred, x_emb, A_pred
