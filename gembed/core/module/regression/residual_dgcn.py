#!/usr/bin/env python3

# TODO: turn into more flexible module
import torch
import torch.nn as nn
from gembed.core.module.spectral import FourierFeatureMap
import torch_geometric.nn as tgnn
from gembed.nn.residual import ResidualCoefficient


class ResidualDGCNRegressionModule(nn.Module):
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
                    "x, batch",
                    [
                        (
                            tgnn.DynamicEdgeConv(
                                nn.Sequential(
                                    nn.Linear(
                                        2 * hidden_dim,
                                        hidden_dim,
                                    ),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                ),
                                k,
                                aggr,
                            ),
                            "x, batch -> x",
                        ),
                        nn.LayerNorm(hidden_dim),
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        ResidualCoefficient(),
                    ],
                )
                for i in range(n_hidden_layers)
            ]
        )

        self.regression = tgnn.Sequential(
            "x, batch",
            [
                # (
                #     #nn.LayerNorm(hidden_dim),
                #     "x -> x",
                # ),  # ADDED v147
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (
                    tgnn.global_max_pool,
                    # tgnn.global_mean_pool,
                    "x, batch -> x",
                ),  # Max aggregation in the feature dimension
                # REGRESSION MODULE
                #nn.LayerNorm(hidden_dim),  # ADDED v147
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                # L2
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                # L3
                nn.Linear(hidden_dim, n_components),
            ],
        )

        # # REGRESSION
        # _regression_module = [
        #     # R1
        #     nn.Linear(hidden_dim, int(hidden_dim / 2)),
        #     nn.ELU(),
        #     # Dropout
        #     # R2
        #     nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
        #     nn.ELU(),
        #     # Dropout
        #     # RF
        #     nn.Linear(int(hidden_dim / 4), n_components),
        # ]

        # if dropout:
        #     _regression_module.insert(2, nn.Dropout(0.1))
        #     _regression_module.insert(5, nn.Dropout(0.1))

        # # final regression layer
        # self.regression = tgnn.Sequential(
        #     "x, batch",
        #     [
        #         (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
        #         (
        #             tgnn.global_max_pool,
        #             "x, batch -> x",
        #         ),  # Max aggregation in the feature dimension
        #         *_regression_module,
        #     ],
        # )
        #

    def feature_forward(self, x, batch):
        x = self.ffm_x(x)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, batch)

        return x

    def forward(self, x, batch):
        x = self.feature_forward(x, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())
