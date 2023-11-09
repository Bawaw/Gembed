# #!/usr/bin/env python3

import torch
import torch.nn as nn
from gembed.core.module.spectral import FourierFeatureMap
import torch_geometric.nn as tgnn
from gembed.nn.residual import ResidualCoefficient


class TransformerBlockRes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=512, k=5):
        super().__init__()
        self.k = k

        # self.pos_nn = nn.Sequential(
        #     nn.Linear(3, hidden_dim),
        #     # nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, out_channels),
        #     # nn.LayerNorm(out_channels), nn.ELU(),
        # )
        # self.pos_nn = nn.Sequential(
        #     FourierFeatureMap(3, hidden_dim, 1.0),
        #     nn.Linear(hidden_dim, out_channels),
        #     # nn.LayerNorm(hidden_dim),
        #     # nn.LayerNorm(out_channels), nn.ELU(),
        # )
        self.pos_nn = nn.Sequential(
            nn.Linear(3, out_channels),
        )
        self.attn_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )

        # self.attn_nn = nn.Sequential(
        #     nn.Linear(out_channels, hidden_dim),
        #     # nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, out_channels),
        #     # nn.LayerNorm(out_channels),
        #     nn.ELU(),
        # )

        self.transformer = tgnn.PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, batch):
        edge_index = tgnn.knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer(x, pos, edge_index)
        return x


class ResidualTransformerRegressionModule(nn.Module):
    def __init__(
        self,
        fourier_feature_scale,
        n_components=512,
        n_hidden_layers=3,
        hidden_dim=512,
        k=20,
        aggr="max",
        dropout=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(3, hidden_dim)
        else:
            self.ffm_x = FourierFeatureMap(3, hidden_dim, fourier_feature_scale)

        input_dim = hidden_dim

        # FEATURE LEARNING
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, pos, batch",
                    [
                        (
                            TransformerBlockRes(hidden_dim, hidden_dim),
                            "x, pos, batch -> x",
                        ),
                        # nn.LayerNorm(hidden_dim),
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        ResidualCoefficient(),
                    ],
                )
                for i in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, batch",
            [
                # (
                #     # nn.LayerNorm(hidden_dim),
                #     "x -> x",
                # ),  # ADDED v147
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (
                    tgnn.global_max_pool,
                    # tgnn.global_mean_pool,
                    "x, batch -> x",
                ),  # Max aggregation in the feature dimension
                # REGRESSION MODULE
                # nn.LayerNorm(hidden_dim),  # ADDED v147
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim), # REMOVED IN V152
                nn.ELU(),
                # L2
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.ELU(),
                # L3
                nn.Linear(hidden_dim, n_components),
            ],
        )

    def feature_forward(self, pos, batch):
        x = self.ffm_x(pos)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, pos, batch)

        return x

    def forward(self, pos, batch):
        x = self.feature_forward(pos, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())
