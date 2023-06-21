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
from gembed.nn.linear import ConcatSquashLinear


class FDyn(nn.Module):
    r""" Models the dynamics."""

    def __init__(self, n_context):
        super().__init__()
        # expected format: N x (C * L)
        # +1 for time
        n_context = n_context + 31  # for time embedding
        self.csl1 = ConcatSquashLinear(512, 512, n_context)
        # self.csl1 = ConcatSquashLinear(3, 512, n_context)
        self.csl2 = ConcatSquashLinear(512, 512, n_context)
        self.csl4 = ConcatSquashLinear(512, 512, n_context)
        self.csl5 = ConcatSquashLinear(512, 3, n_context)

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        self.scale = 2 ** -2
        self.Wx = nn.Parameter(
            self.scale * torch.randn(512 // 2, 3), requires_grad=False
        )

        self.scale_t = 2 ** -2
        self.Wt = nn.Parameter(
            self.scale_t * torch.randn(32 // 2, 1), requires_grad=False
        )

        print(f"Running FDYN with wx scale: {self.scale}")
        print(f"Running FDYN with tx scale: {self.scale_t}")

    def gaussian_encoding(self, v, b):
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
            b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`
        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`
        See :class:`~rff.layers.GaussianEncoding` for more details.

        Source: https://github.com/jmclong/random-fourier-features-pytorch/tree/main/rff
        """
        vp = 2 * np.pi * v @ b.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

    def forward(self, t, x, c, **kwargs):
        x = self.gaussian_encoding(x, self.Wx)
        t = self.gaussian_encoding(t.unsqueeze(0), self.Wt)

        context = torch.concat([c, t.repeat([x.shape[0], 1])], -1)

        x = torch.tanh(self.csl1(context, x))
        x = torch.tanh(self.csl2(context, x))
        x = torch.tanh(self.csl4(context, x))
        x = self.csl5(context, x)
        return x


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn):
        super().__init__()
        self.feature_nn = feature_nn

    def inverse(self, pos, batch):
        return self.feature_nn(pos, batch)

    def __str__(self):
        return str(self.__class__.str())


class RegularisedPointFlowSTNModel(RegularisedPointFlowSTN):
    def __init__(
        self,
        n_components=128,
        k=20,
        aggr="max",
        integration_method="rk4",
        adjoint=False,
    ):
        # point distribution model
        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
            layers=RegularisedContinuousAmbientFlow(
                FDyn(n_context=n_components),
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=True,
                method=integration_method,
                options=None,
                adjoint=adjoint,
            ),
        )

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    # INPUT LAYER
                    # (nn.Dropout(0.2), "pos -> pos"),
                    # FEATURE BLOCK 1
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    # FEATURE BLOCK 2
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    # FEATURE BLOCK 3
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    # AGGREGATION BLOCK
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 128, 1024),
                    (tgnn.global_max_pool, "x, batch -> x"),
                    # REGRESSION BLOCK
                    nn.Sequential(
                        # R1
                        nn.Linear(1024, 512),
                        nn.ELU(),
                        nn.Dropout(0.1),
                        # R2
                        nn.Linear(512, 256),
                        nn.ELU(),
                        nn.Dropout(0.1),
                        # RF
                        nn.Linear(256, n_components),
                    ),
                ],
            )
        )

        stn = SpatialTransformer(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    # (nn.Dropout(0.2), "pos -> pos"),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 64, 64),
                    (tgnn.global_max_pool, "x, batch -> x"),
                ],
            ),
            regressor_nn=nn.Sequential(
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 2 * 3),
            ),
        )

        super().__init__(sdm=sdm, pdm=pdm, stn=stn)


class RegularisedPointFlowSTNMLNModel(RegularisedPointFlowSTN):
    def __init__(
        self,
        n_components=128,
        k=20,
        aggr="max",
        integration_method="rk4",
        adjoint=False,
        mln_metric="euclidean",
    ):
        # point distribution model
        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
            layers=RegularisedContinuousAmbientFlow(
                FDyn(n_context=n_components),
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=True,
                method=integration_method,
                options=None,
                adjoint=adjoint,
            ),
        )

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    # INPUT LAYER
                    # (nn.Dropout(0.2), "pos -> pos"),
                    # FEATURE BLOCK 1
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    # FEATURE BLOCK 2
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    # FEATURE BLOCK 3
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                                nn.ELU(),
                                nn.Linear(128, 128),
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    # AGGREGATION BLOCK
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 128, 1024),
                    (tgnn.global_max_pool, "x, batch -> x"),
                    # Manifold Learning Module
                    (
                        ContinuousDGM(
                            embed_f=nn.Sequential(
                                nn.Linear(1024, 512),
                                nn.ELU(),
                                nn.Dropout(0.1),
                                nn.Linear(512, 256),
                                nn.ELU(),
                                nn.Dropout(0.1),
                                nn.Linear(256, n_components),
                            ),
                            distance=mln_metric,
                            input_dim=n_components,
                        ),
                        "x -> x, edge_index, edge_weight",
                    ),
                    (
                        tgnn.GCNConv(n_components, n_components),
                        "x, edge_index, edge_weight -> x",
                    ),
                ],
            )
        )

        stn = SpatialTransformer(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    # (nn.Dropout(0.2), "pos -> pos"),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU()  # , nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 64, 64),
                    (tgnn.global_max_pool, "x, batch -> x"),
                ],
            ),
            regressor_nn=nn.Sequential(
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 2 * 3),
            ),
        )

        super().__init__(sdm=sdm, pdm=pdm, stn=stn)


def point_flow_model(pretrained=False, progress=True, **kwargs):
    model = PointFlowModel(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
    #     model.load_state_dict(state_dict)
    return model
