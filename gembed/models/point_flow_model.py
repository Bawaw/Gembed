#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.mln import ContinuousDGM
from gembed.core.distribution import MultivariateNormal
from gembed.core.module import (
    InvertibleModule,
    NormalisingFlow,
    # SingleLaneAugmentedPointFlow,
    # RegularisedSingleLaneAugmentedPointFlow,
    RegularisedPointFlowSTN,
)
from gembed.core.module.bijection import (
    ContinuousAmbientFlow,
    RegularisedContinuousAmbientFlow,
)
from gembed.nn.linear.concat_squash_linear import *
from gembed.core.module.spectral import FourierFeatureMap


class LinearCombination(nn.Module):
    def __init__(self, include_bias=False, *dims):
        super().__init__()

        dim_in = dims[:-1]
        dim_out = dims[-1]

        self.layers = nn.ModuleList()
        for dim_in in dim_in:
            self.layers.append(nn.Linear(dim_in, dim_out, bias=False))

            # orthogonal initialisation
            # _layer = nn.Linear(dim_in, dim_out, bias=False)
            # torch.nn.init.orthogonal_(_layer.weight)
            # self.layers.append(_layer)

        self.include_bias = include_bias

        if include_bias:
            self.bias = nn.parameter.Parameter(
                data=torch.empty(dim_out), requires_grad=True
            )
            self.bias.data.zero_()

    def forward(self, *inputs):
        res = self.layers[0](inputs[0])

        # W1x1 + W2x2 + ... + Wnxn
        for i in range(1, len(inputs)):
            res += self.layers[i](inputs[i])

        if self.include_bias:
            res += self.bias

        return res


class HyperCombination(nn.Module):
    def __init__(self, in_dim, out_dim, context_dim, hidden_dim, include_bias=False):
        super().__init__()

        self.in_dim, self.out_dim, self.context_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            context_dim,
            hidden_dim,
        )
        # self.hyper_net = nn.Linear(512,  512 * (512 + 32), bias=False)
        self.a_shape = self.out_dim
        self.W_shape = (self.out_dim, self.in_dim)

        self.hyper_net = nn.Sequential(
            # nn.Linear(2 * 3, 128),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.a_shape + math.prod(self.W_shape)),
        )

    def forward(self, x, c, t):
        weights = self.hyper_net(c[:1])

        x = torch.cat([x, t.repeat(x.shape[0], 1)], 1)
        a = weights[:, : self.a_shape]
        W = weights[:, self.a_shape :].view(*self.W_shape)

        return a * (x @ W.T)


# class HyperLinear(nn.Module):
#     def __init__(self, include_bias=False, *dims):
#         super().__init__()

#         dim_in = dims[:-1]
#         dim_out = dims[-1]

#         # self.hyper_net = nn.Linear(512,  512 * (512 + 32), bias=False)
#         self.out_channels = dim_out

#         self.hyper_net = nn.Sequential(
#             # nn.Linear(2 * 3, 128),
#             # nn.LayerNorm(512),
#             nn.Linear(512, 512),
#             nn.Softplus(),
#             nn.Linear(512, 512),
#             nn.Softplus(),
#             nn.Linear(512, 2 * self.out_channels + 512 * self.out_channels),
#         )

#     def forward(self, x, c, t):
#         weights = self.hyper_net(c[:1])

#         a, b, W = (
#             weights[:, : self.out_channels],
#             weights[:, self.out_channels : 2 * self.out_channels],
#             weights[:, 2 * self.out_channels :].view(self.out_channels, 512),
#         )
#         return a * torch.nn.functional.linear(x, weight=W, bias=b)


class HyperConcatSquashLinear(nn.Module):
    def __init__(self, in_dim, out_dim, context_dim, hidden_dim, include_bias=False):
        super().__init__()

        self.in_dim, self.out_dim, self.context_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            context_dim,
            hidden_dim,
        )
        self.W_shape = (self.out_dim, self.in_dim)

        # self.hyper_bias = nn.Linear(32, self.out_dim, bias=False)
        self.hyper_bias = nn.Sequential(
            # nn.Linear(2 * 3, 128),
            nn.LayerNorm(32),  # ADDED in V147
            nn.Linear(32, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        # self.hyper_gate = nn.Linear(32, self.out_dim)
        self.hyper_gate = nn.Sequential(
            # nn.Linear(2 * 3, 128),
            nn.LayerNorm(32),  # ADDED in V147
            nn.Linear(32, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        self.hyper_net = nn.Sequential(
            # nn.Linear(2 * 3, 128),
            nn.LayerNorm(self.context_dim),  # ADDED in V147
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Softplus(),
            nn.Linear(
                self.hidden_dim,
                math.prod(self.W_shape),
            ),
        )

    def forward(self, x, c, t):
        W = self.hyper_net(c[:1]).view(self.W_shape)
        a = torch.sigmoid(self.hyper_gate(t.view(1, 32)))
        b = self.hyper_bias(t.view(1, 32))

        return (a * (x @ W.T)) + b


class HyperLinear(nn.Module):
    def __init__(self, in_dim, out_dim, context_dim, hidden_dim, include_bias=False):
        super().__init__()

        self.in_dim, self.out_dim, self.context_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            context_dim,
            hidden_dim,
        )

        self.a_shape = self.out_dim
        self.b_shape = self.out_dim
        self.W_shape = (self.out_dim, self.in_dim)

        self.hyper_net = nn.Sequential(
            # nn.Linear(2 * 3, 128),
            # nn.LayerNorm(512),
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(
                self.hidden_dim,
                self.b_shape + self.a_shape + math.prod(self.W_shape),
            ),
        )

    def forward(self, x, c, t):
        weights = self.hyper_net(c[:1])

        b = weights[:, : self.a_shape]
        a = weights[:, self.a_shape : 2 * self.a_shape]
        W = weights[:, 2 * self.a_shape :].view(*self.W_shape)

        return (a * (x @ W.T)) + b


class FuseCombination(nn.Module):
    def __init__(self, include_bias=False, *dims):
        super().__init__()

        dim_in = dims[:-1]
        dim_out = dims[-1]

        self.layers = nn.Linear(sum(dim_in), dim_out, bias=False)

    def forward(self, x, c, t):
        return self.layers(torch.cat([x, c, t.repeat(x.shape[0], 1)], 1))


class ResidualCoefficient(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = torch.nn.parameter.Parameter(
            data=torch.zeros(1), requires_grad=True
        )

    def forward(self, x):
        return self.alpha * x


class FDyn(nn.Module):
    r""" Models the dynamics."""

    def __init__(
        self,
        n_context,
        fourier_feature_scale,
        in_channels=3,
        n_hidden_layers=3,
        hyper_hidden_dim=512,
        hidden_dim=512,
        out_channels=3,
    ):
        super().__init__()

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        print(f"Fourier feature scale: {fourier_feature_scale}")
        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(in_channels, hidden_dim, fourier_feature_scale)
            self.ffm_t = nn.Linear(1, 32, fourier_feature_scale)
        else:
            self.ffm_x = FourierFeatureMap(
                in_channels, hidden_dim, fourier_feature_scale
            )
            self.ffm_t = FourierFeatureMap(1, 32, fourier_feature_scale)

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, c, t",
                    [
                        (nn.LayerNorm(hidden_dim), "x -> x"),  # ADDED v147
                        (
                            # LinearCombination(
                            #     False, hidden_dim, n_context, 32, hidden_dim
                            # ),
                            # HyperCombination(
                            #     hidden_dim + 32,
                            #     hidden_dim,
                            #     n_context,
                            #     hyper_hidden_dim,
                            #     False,
                            # ),
                            HyperConcatSquashLinear(
                                hidden_dim,
                                hidden_dim,
                                n_context,
                                hyper_hidden_dim,
                                True,
                            ),
                            # FuseCombination(
                            #     False, hidden_dim, n_context, 32, hidden_dim
                            # ),
                            "x, c, t -> x",
                        ),
                        nn.LayerNorm(hidden_dim),
                        nn.Softplus(),
                        # nn.Linear(hidden_dim, hidden_dim),
                        (
                            HyperConcatSquashLinear(
                                hidden_dim,
                                hidden_dim,
                                n_context,
                                hyper_hidden_dim,
                                True,
                            ),
                            # HyperLinear(
                            #     hidden_dim,
                            #     hidden_dim,
                            #     n_context,
                            #     hyper_hidden_dim,
                            #     True,
                            # ),
                            "x, c, t -> x",
                        ),
                        ResidualCoefficient(),
                    ],
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, c, t",
            [
                # (
                #     nn.Linear(hidden_dim, out_channels),
                #     "x -> x",
                # ),
                # (
                #     HyperLinear(
                #         hidden_dim, hidden_dim, n_context, hyper_hidden_dim, True
                #     ),
                #     "x,c,t -> x",
                # ),
                # nn.Softplus(),
                # (
                #     HyperLinear(hidden_dim, 3, n_context, hyper_hidden_dim, True),
                #     "x,c,t -> x",
                # ),
                (nn.LayerNorm(hidden_dim), "x -> x"),  # ADDED v147
                (
                    HyperConcatSquashLinear(
                        hidden_dim, hidden_dim, n_context, hyper_hidden_dim, True
                    ),
                    "x,c,t -> x",
                ),
                nn.Softplus(),
                (
                    HyperConcatSquashLinear(
                        hidden_dim, 3, n_context, hyper_hidden_dim, True
                    ),
                    "x,c,t -> x",
                ),
                ResidualCoefficient(),
            ],
        )

        # from copy import deepcopy

        # self.layers_2 = deepcopy(self.layers)

    def forward(self, t, x, c, **kwargs):
        # prep input
        x = self.ffm_x(x)
        t = self.ffm_t(t.unsqueeze(0))[None]

        for f in self.layers:
            x = x + f(x, c, t)

        # return velocity
        return self.regression(x, c, t)


import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    PointTransformerConv,
    fps,
    global_mean_pool,
    global_max_pool,
    knn,
    knn_graph,
)
from torch_scatter import scatter_mean, scatter_sum, scatter_max


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=512):
        super().__init__()
        # self.lin_in = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        # )
        # self.lin_out = nn.Sequential(
        #     # nn.Linear(out_channels, out_channels),
        #     nn.LayerNorm(),
        #     nn.ELU(),
        # )
        # self.lin_in = nn.Linear(in_channels, in_channels)
        # self.lin_out = nn.Linear(out_channels, out_channels)

        # self.pos_nn = nn.Sequential(
        #     nn.Linear(3, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, out_channels),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        # )

        # self.attn_nn = nn.Sequential(
        #     nn.Linear(out_channels, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        #     nn.Linear(hidden_dim, out_channels),
        #     nn.LayerNorm(hidden_dim),
        #     nn.ELU(),
        # )
        self.pos_nn = None
        self.attn_nn = None

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        # x = self.lin_in(x)
        x = self.transformer(x, pos, edge_index)
        # x = self.lin_out(x)
        return x


class TransitionDown(torch.nn.Module):
    """
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality
    """

    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(
            pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
        )

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter_max(
            x[id_k_neighbor[1]],
            id_k_neighbor[0],
            dim=0,
            dim_size=id_clusters.size(0),
        )[0]

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class TransformerRegressionModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, fourier_feature_scale, dim_model, k=20
    ):
        super().__init__()
        self.k = k

        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(in_channels, dim_model[0])
        else:
            self.ffm_x = FourierFeatureMap(
                in_channels, dim_model[0], fourier_feature_scale
            )

        # FEATURE LEARNING
        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0], out_channels=dim_model[0]
        )

        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

        # self.mlp_output = nn.Sequential(
        #     nn.Linear(dim_model[-1], 64),
        #     nn.ELU(),
        #     nn.Linear(64, out_channels),
        # )
        self.mlp_output = nn.Sequential(
            # L1
            nn.Linear(dim_model[-1], dim_model[-1]),
            nn.LayerNorm(dim_model[-1]),
            nn.ELU(),
            # L1
            nn.Linear(dim_model[-1], dim_model[-1]),
            nn.LayerNorm(dim_model[-1]),
            nn.ELU(),
            # L3
            nn.Linear(dim_model[-1], out_channels),
        )

    def forward(self, pos, batch):
        x = self.ffm_x(pos)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        x = global_max_pool(x, batch)

        out = self.mlp_output(x)
        # return condition
        return out

    def __str__(self):
        return str(self.__class__.str())


class TransformerBlockRes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=512, k=20):
        super().__init__()
        self.k = k

        self.pos_nn = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
        )

        self.attn_nn = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.ELU(),
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, k=self.k, batch=batch)
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
                        nn.LayerNorm(hidden_dim),
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
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (
                    tgnn.global_max_pool,
                    "x, batch -> x",
                ),  # Max aggregation in the feature dimension
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                # L3
                nn.Linear(hidden_dim, n_components),
            ],
        )

    def forward(self, pos, batch):
        x = self.ffm_x(pos)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, pos, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())


class DGCNRegressionModule(nn.Module):
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

        # REGRESSION
        _regression_module = [
            # R1
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.ELU(),
            # Dropout
            # R2
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)),
            nn.ELU(),
            # Dropout
            # RF
            nn.Linear(int(hidden_dim / 4), n_components),
        ]

        if dropout:
            _regression_module.insert(2, nn.Dropout(0.1))
            _regression_module.insert(5, nn.Dropout(0.1))

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, batch",
            [
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (
                    tgnn.global_max_pool,
                    "x, batch -> x",
                ),  # Max aggregation in the feature dimension
                *_regression_module,
            ],
        )

    def forward(self, x, batch):
        x = self.ffm_x(x)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn):
        super().__init__()
        self.feature_nn = feature_nn

    def inverse(self, pos, batch):
        return self.feature_nn(pos, batch)

    def __str__(self):
        return str(self.__class__.str())


class FCRegressionModule(nn.Module):
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

        # FEATURE LEARNING
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, batch",
                    [
                        (
                            nn.LayerNorm(hidden_dim),  # ADDED v147
                            "x -> x",
                        ),
                        (
                            nn.Linear(hidden_dim, hidden_dim),
                            "x -> x",
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

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, batch",
            [
                (
                    nn.LayerNorm(hidden_dim),
                    "x -> x",
                ),  # ADDED v147
                (nn.Linear(hidden_dim, hidden_dim), "x -> x"),
                (
                    tgnn.global_max_pool,
                    # tgnn.global_mean_pool,
                    "x, batch -> x",
                ),  # Max aggregation in the feature dimension
                # REGRESSION MODULE
                nn.LayerNorm(hidden_dim),  # ADDED v147
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim), # REMOVED IN V152
                nn.ELU(),
                # L1
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.ELU(),
                # L3
                nn.Linear(hidden_dim, n_components),
            ],
        )

    def forward(self, x, batch):
        x = self.ffm_x(x)

        # Compute hidden representation with residual connections
        for i, f in enumerate(self.layers):
            x = x + f(x, batch)

        # return condition
        return self.regression(x, batch)

    def __str__(self):
        return str(self.__class__.str())


# class RegularisedPointFlowSTNModel(RegularisedPointFlowSTN):
#     def __init__(
#         self,
#         n_components=128,
#         k=20,
#         aggr="max",
#         integration_method="rk4",
#         fourier_feature_scale=1.0,
#         adjoint=False,
#     ):

#         # spatial transformer
#         stn = SpatialTransformer(
#             DGCNRegressionModule(
#                 n_components=2 * 3,
#                 fourier_feature_scale=-1,
#                 n_hidden_layers=1,
#                 hidden_dim=64,
#                 dropout=False,
#             )
#         )

#         # shape distribution model
#         # print("Warning! no SDM dropout!")
#         sdm = ShapeModel(
#             DGCNRegressionModule(
#                 n_components=n_components,
#                 fourier_feature_scale=fourier_feature_scale,
#                 n_hidden_layers=2,
#                 hidden_dim=512,
#                 dropout=False,
#             )
#         )

#         # point distribution model
#         fdyn = FDyn(
#             n_context=n_components,
#             fourier_feature_scale=fourier_feature_scale,
#             in_channels=4,
#             out_channels=3,
#             n_hidden_layers=3,
#             hidden_dim=512,
#         )
#         adyn = FDyn(
#             n_context=n_components,
#             fourier_feature_scale=fourier_feature_scale,
#             in_channels=1,
#             out_channels=1,
#             n_hidden_layers=1,
#             hidden_dim=512,
#         )

#         import gembed.core.module.bijection.augmented_regularised_continuous_ambient_flow as arcaf

#         pdm = NormalisingFlow(
#             base_distribution=MultivariateNormal(
#                 # torch.zeros(3), torch.Tensor([1/3,1/3,1e-5]) * torch.eye(3, 3)
#                 torch.zeros(3),
#                 (1 / 3) * torch.eye(3, 3),
#             ),
#             layers=arcaf.AugmentedRegularisedContinuousAmbientFlow(
#                 fdyn,
#                 adyn,
#                 noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#                 estimate_trace=True,
#                 method=integration_method,
#                 adjoint=adjoint,
#             ),
#         )

#         super().__init__(sdm=sdm, pdm=pdm, stn=stn)


class RegularisedPointFlowSTNModel(RegularisedPointFlowSTN):
    def __init__(
        self,
        n_components=128,
        k=20,
        aggr="max",
        integration_method="rk4",
        fourier_feature_scale=1.0,
        adjoint=False,
    ):
        # NETWORK CONFIG

        # spatial transformer
        # stn = None
        stn = SpatialTransformer(
            DGCNRegressionModule(
                n_components=2 * 3,
                fourier_feature_scale=-1,
                n_hidden_layers=1,
                hidden_dim=64,
                dropout=False,
            )
        )

        # shape distribution model
        # print("Warning! no SDM!")
        # sdm = None
        sdm = ShapeModel(
            # DGCNRegressionModule(
            #     n_components=n_components,
            #     fourier_feature_scale=fourier_feature_scale,
            #     n_hidden_layers=1,
            #     hidden_dim=512,
            #     dropout=False,
            # )
            # TransformerRegressionModule(
            #     in_channels=3,
            #     out_channels=n_components,
            #     dim_model=[32, 64, 128, 256, 512],
            #     # dim_model=[32, 64, 128],
            #     # dim_model=5 * [512],
            #     # dim_model=[512],
            #     fourier_feature_scale=fourier_feature_scale,
            # )
            FCRegressionModule(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale,
                n_hidden_layers=1,
                hidden_dim=64,
                dropout=False,
            )
            # ResidualTransformerRegressionModule(
            #     n_components=n_components,
            #     fourier_feature_scale=fourier_feature_scale,
            #     n_hidden_layers=0,
            #     hidden_dim=64,
            #     dropout=False,
            # )
        )

        # point distribution model
        fdyn = FDyn(
            n_context=n_components,
            fourier_feature_scale=fourier_feature_scale,
            n_hidden_layers=1,
            hidden_dim=64,
            hyper_hidden_dim=128,
        )

        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(
                torch.zeros(3),
                (1 / 3) * torch.eye(3, 3),
            ),
            layers=RegularisedContinuousAmbientFlow(
                fdyn,
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=True,
                method=integration_method,
                adjoint=adjoint,
            ),
        )

        super().__init__(sdm=sdm, pdm=pdm, stn=stn)


# class RegularisedPointFlowSTNModel(RegularisedPointFlowSTN):
#     def __init__(
#         self,
#         n_components=128,
#         k=20,
#         aggr="max",
#         integration_method="rk4",
#         fourier_feature_scale=None,
#         adjoint=False,
#     ):
#         if fourier_feature_scale is None:
#             fourier_feature_scale = 2 ** -2

#         # point distribution model
#         pdm = NormalisingFlow(
#             base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#             layers=RegularisedContinuousAmbientFlow(
#                 FDyn(
#                     n_context=n_components, fourier_feature_scale=fourier_feature_scale
#                 ),
#                 noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#                 estimate_trace=True,
#                 method=integration_method,
#                 adjoint=adjoint,
#             ),
#         )

#         # shape distribution model
#         # sdm = None
#         sdm = ShapeModel(
#             feature_nn=tgnn.Sequential(
#                 "pos, batch",
#                 [
#                     (
#                         FourierFeatureMap(3, 128, fourier_feature_scale),
#                         "pos -> pos",
#                     ),
#                     # INPUT LAYER
#                     # (nn.Dropout(0.2), "pos -> pos"),
#                     # FEATURE BLOCK 1
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 # nn.Linear(2 * 3, 128),
#                                 nn.Linear(2 * 128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "pos, batch -> x1",
#                     ),
#                     # FEATURE BLOCK 2
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x1, batch -> x2",
#                     ),
#                     # FEATURE BLOCK 3
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x2, batch -> x3",
#                     ),
#                     # AGGREGATION BLOCK
#                     (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     nn.Linear(3 * 128, 1024),
#                     (tgnn.global_max_pool, "x, batch -> x"),
#                     # REGRESSION BLOCK
#                     nn.Sequential(
#                         # R1
#                         nn.Linear(1024, 512),
#                         nn.ELU(),
#                         nn.Dropout(0.1),
#                         # R2
#                         nn.Linear(512, 256),
#                         nn.ELU(),
#                         nn.Dropout(0.1),
#                         # RF
#                         nn.Linear(256, n_components),
#                     ),
#                 ],
#             )
#         )

#         # stn = None
#         stn = SpatialTransformer(
#             feature_nn=tgnn.Sequential(
#                 "pos, batch",
#                 [
#                     # (
#                     #     FourierFeatureMap(3, 64, 0.1),
#                     #     "pos -> pos",
#                     # ),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 3, 64),
#                                 nn.ELU(),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "pos, batch -> x1",
#                     ),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(nn.Linear(2 * 64, 64), nn.ELU()),
#                             k,
#                             aggr,
#                         ),
#                         "x1, batch -> x2",
#                     ),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(nn.Linear(2 * 64, 64), nn.ELU()),
#                             k,
#                             aggr,
#                         ),
#                         "x2, batch -> x3",
#                     ),
#                     (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     nn.Linear(3 * 64, 64),
#                     (tgnn.global_max_pool, "x, batch -> x"),
#                 ],
#             ),
#             regressor_nn=nn.Sequential(
#                 nn.Linear(64, 32),
#                 nn.ELU(),
#                 nn.Linear(32, 2 * 3),
#                 # R1
#                 # nn.Linear(64, 32),
#                 # nn.ELU(),
#                 # nn.Dropout(0.1),
#                 # # R2
#                 # nn.Linear(32, 32),
#                 # nn.ELU(),
#                 # nn.Dropout(0.1),
#                 # # RF
#                 # nn.Linear(32, 2 * 3),
#             ),
#         )

#         super().__init__(sdm=sdm, pdm=pdm, stn=stn)


# class RegularisedPointFlowSTNMLNModel(RegularisedPointFlowSTN):
#     def __init__(
#         self,
#         n_components=128,
#         k=20,
#         aggr="max",
#         integration_method="rk4",
#         adjoint=False,
#         mln_metric="euclidean",
#     ):
#         # point distribution model
#         pdm = NormalisingFlow(
#             base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#             layers=RegularisedContinuousAmbientFlow(
#                 FDyn(n_context=n_components),
#                 noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#                 estimate_trace=True,
#                 method=integration_method,
#                 options=None,
#                 adjoint=adjoint,
#             ),
#         )

#         # shape distribution model
#         sdm = ShapeModel(
#             feature_nn=tgnn.Sequential(
#                 "pos, batch",
#                 [
#                     # INPUT LAYER
#                     # (nn.Dropout(0.2), "pos -> pos"),
#                     # FEATURE BLOCK 1
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 3, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "pos, batch -> x1",
#                     ),
#                     # FEATURE BLOCK 2
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x1, batch -> x2",
#                     ),
#                     # FEATURE BLOCK 3
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                                 nn.ELU(),
#                                 nn.Linear(128, 128),
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x2, batch -> x3",
#                     ),
#                     # AGGREGATION BLOCK
#                     (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     nn.Linear(3 * 128, 1024),
#                     (tgnn.global_max_pool, "x, batch -> x"),
#                     # Manifold Learning Module
#                     (
#                         # ContinuousDGM(
#                         #     embed_f=nn.Sequential(
#                         #         nn.Linear(1024, 512),
#                         #         nn.ELU(),
#                         #         nn.Dropout(0.1),
#                         #         nn.Linear(512, 256),
#                         #         nn.ELU(),
#                         #         nn.Dropout(0.1),
#                         #         nn.Linear(256, n_components),
#                         #     ),
#                         #     distance=mln_metric,
#                         #     input_dim=n_components,
#                         # ),
#                         # "x -> x, edge_index, edge_weight",
#                         #
#                         # tgnn.GCNConv(n_components, n_components),
#                         # "x, edge_index, edge_weight -> x",
#                         ContinuousDGM(
#                             embed_f=nn.Sequential(
#                                 nn.Linear(1024, 512),
#                                 nn.ELU(),
#                                 #nn.Dropout(0.1),
#                                 nn.Linear(512, 256),
#                                 nn.ELU(),
#                                 #nn.Dropout(0.1),
#                                 nn.Linear(256, n_components),
#                             ),
#                             distance=mln_metric,
#                             input_dim=n_components,
#                         ),
#                         # "x -> x, edge_index, edge_weight",
#                         "x -> x",
#                     ),
#                     # (
#                     #     tgnn.GCNConv(n_components, n_components),
#                     #     "x, edge_index, edge_weight -> x",
#                     #     #     tgnn.SimpleConv(aggr="mean", combine_root="self_loop"),
#                     #     #     "x, edge_index, edge_weight -> x",
#                     # ),
#                 ],
#             )
#         )
#         print("WARNING: NO DROPOUT")

#         stn = SpatialTransformer(
#             feature_nn=tgnn.Sequential(
#                 "pos, batch",
#                 [
#                     # (nn.Dropout(0.2), "pos -> pos"),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 3, 64), nn.ELU()  # , nn.BatchNorm1d(64)
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "pos, batch -> x1",
#                     ),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x1, batch -> x2",
#                     ),
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
#                             ),
#                             k,
#                             aggr,
#                         ),
#                         "x2, batch -> x3",
#                     ),
#                     (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     nn.Linear(3 * 64, 64),
#                     (tgnn.global_max_pool, "x, batch -> x"),
#                 ],
#             ),
#             regressor_nn=nn.Sequential(
#                 nn.Linear(64, 32),
#                 nn.ELU(),
#                 nn.Linear(32, 2 * 3),
#             ),
#         )

#         super().__init__(sdm=sdm, pdm=pdm, stn=stn)


def point_flow_model(pretrained=False, progress=True, **kwargs):
    model = PointFlowModel(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
    #     model.load_state_dict(state_dict)
    return model
