#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

from gembed.core.module.nn.residual import ResidualCoefficient
from gembed.core.module.nn.spectral import FourierFeatureMap


class TransformerBlockRes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k, hidden_dim=512):
        super().__init__()
        self.k = k

        self.pos_nn = nn.Linear(3, out_channels)
        self.attn_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )

        self.transformer = tgnn.PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, batch):
        edge_index = tgnn.knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer(x, pos, edge_index)
        return x


class ResidualPCEncoder(nn.Module):
    """The `ResidualPCEncoder` class is a PyTorch module that implements a residual point cloud encoder
    with various layer types and activation functions.
    """
    def __init__(
        self,
        fourier_feature_scale,
        n_components,
        n_hidden_layers,
        hidden_dim,
        layer_type,
        activation_type,
        **layer_kwargs
    ):
        super().__init__()

        self.layer_type = layer_type
        self.activation_type = activation_type
        self.hidden_dim = hidden_dim

        if fourier_feature_scale is None:
            self.ffm_x = nn.Linear(3, hidden_dim)
        else:
            self.ffm_x = FourierFeatureMap(3, hidden_dim, fourier_feature_scale)

        def layer():
            if self.layer_type == "dgcnn":
                if not "k" in layer_kwargs:
                    layer_kwargs["k"] = 20

                return (
                    tgnn.DynamicEdgeConv(
                        nn.Linear(2 * self.hidden_dim, self.hidden_dim), **layer_kwargs
                    ),
                    "x, batch -> x",
                )

            if self.layer_type == "pointnet":
                return (nn.Linear(hidden_dim, hidden_dim), "x -> x")

            if self.layer_type == "transformer":
                if not "k" in layer_kwargs:
                    layer_kwargs["k"] = 20
                return (
                    TransformerBlockRes(hidden_dim, hidden_dim, **layer_kwargs),
                    "x, pos, batch -> x",
                )

        def activation():
            if self.activation_type == "tanh":
                return nn.Tanh()
            elif self.activation_type == "tanhshrink":
                return nn.Tanhshrink()
            elif self.activation_type == "softplus":
                return nn.Softplus()
            elif self.activation_type == "swish":
                return nn.SiLU()
            elif self.activation_type == "elu":
                return nn.ELU()
            elif self.activation_type == "relu":
                return nn.ReLU()
            else:
                assert False

        # FEATURE LEARNING
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, pos, batch",
                    [
                        layer(),
                        nn.BatchNorm1d(hidden_dim),
                        activation(),
                        nn.Linear(hidden_dim, hidden_dim),
                        ResidualCoefficient(),
                    ],
                )
                for i in range(n_hidden_layers)
            ]
        )

        self.global_pool = tgnn.Sequential(
            "x, batch",
            [
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (tgnn.global_max_pool, "x, batch -> x"),
            ],
        )

        # final regression layer
        self.regression = nn.Sequential(
            # REGRESSION MODULE
            # L1
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # L2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # L3
            nn.Linear(hidden_dim, n_components),
        )

    def feature_forward(self, pos, batch):
        x = self.ffm_x(pos)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, pos, batch)

        x = self.global_pool(x, batch)

        return x

    def forward(self, x, batch):
        x = self.feature_forward(x, batch)

        # return condition
        return self.regression(x)

    def __str__(self):
        return str(self.__class__.str())
