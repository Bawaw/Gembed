#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

from gembed.core.module.nn.residual import ResidualCoefficient
from gembed.core.module.nn.spectral import FourierFeatureMap

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
        global_pool_dim,
        mlp_dim,
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

        self.nn_x = nn.Linear(hidden_dim, hidden_dim)

        def layer():
            if self.layer_type == "dgcnn":
                if not "k" in layer_kwargs:
                    layer_kwargs["k"] = 14

                return (
                    tgnn.DynamicEdgeConv(
                        nn.Linear(2 * self.hidden_dim, self.hidden_dim), **layer_kwargs
                    ),
                    "x, batch -> x",
                )

            if self.layer_type == "pointnet":
                return (nn.Linear(hidden_dim, hidden_dim), "x -> x")

        def activation():
            if self.activation_type == "sin":
                return torch.sin
            elif self.activation_type == "selu":
                return nn.SELU()
            elif self.activation_type == "tanh":
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
                        (nn.BatchNorm1d(hidden_dim), "x -> x"),
                        activation(),
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
                (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(), 
                nn.Linear(hidden_dim, global_pool_dim),
                (tgnn.global_max_pool, "x, batch -> x"),
            ],
        )

        # final regression layer
        self.regression = nn.Sequential(
            # REGRESSION MODULE
            # L1
            nn.Linear(global_pool_dim, mlp_dim[0]),
            activation(),
            # L2
            nn.Linear(mlp_dim[0], mlp_dim[1]),
            activation(),
            # L3
            nn.Linear(mlp_dim[1], n_components),
        )

    def feature_forward(self, pos, batch, apply_global_pooling=True):
        x = self.ffm_x(pos)

        for i, f in enumerate(self.layers):
            if i == 0: 
                x = x + f(self.nn_x(x), pos, batch)
            else:
                x = x + f(x, pos, batch)

        if apply_global_pooling:
            x = self.global_pool(x, batch)

        return x

    def forward(self, x, batch):
        x = self.feature_forward(x, batch)

        # return shape representation
        return self.regression(x)

    def __str__(self):
        return str(self.__class__.str())
