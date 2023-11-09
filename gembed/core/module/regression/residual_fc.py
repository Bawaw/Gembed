#!/usr/bin/env python3


import torch.nn.functional as F
import torch.nn as nn
from gembed.core.module.spectral import FourierFeatureMap
import torch_geometric.nn as tgnn
from gembed.nn.residual import ResidualCoefficient


class ResidualFCRegressionModule(nn.Module):
    def __init__(
        self,
        fourier_feature_scale,
        n_components=512,
        n_hidden_layers=3,
        hidden_dim=512,
        k=20,
        aggr="max",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(3, hidden_dim)
        else:
            self.ffm_x = FourierFeatureMap(3, hidden_dim, fourier_feature_scale)

        activation_type = "elu"

        def activation():
            if activation_type == "tanh":
                return nn.Tanh()
            elif activation_type == "tanhshrink":
                return nn.Tanhshrink()
            elif activation_type == "softplus":
                return nn.Softplus()
            elif activation_type == "swish":
                return nn.SiLU()
            elif activation_type == "elu":
                return nn.ELU()
            else:
                assert False

        # FEATURE LEARNING
        self.layers = nn.ModuleList(
             [
                tgnn.Sequential(
                    "x, batch",
                    [
                        (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                        nn.LayerNorm(hidden_dim),
                        activation(),
                        nn.Linear(hidden_dim, hidden_dim),
                        #nn.LayerNorm(hidden_dim),
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
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (tgnn.global_max_pool, "x, batch -> x"),

                # REGRESSION MODULE
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation(),
                # L2
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                # L3
                nn.Linear(hidden_dim, n_components),
            ],
        )

    def feature_forward(self, x, batch):
        x = self.ffm_x(x)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, batch)
            #x = F.elu(x + f(x, batch))

        return x

    def forward(self, x, batch):
        x = self.feature_forward(x, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())
