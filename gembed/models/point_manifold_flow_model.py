#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.mln import ContinuousDGM
from gembed.core.distribution import MultivariateNormal
from gembed.nn.fusion import HyperConcatSquash
from gembed.core.module import (
    InvertibleModule,
    NormalisingFlow,
    RegularisedPointFlowSTN,
    RegularisedPointManifoldFlowSTN,
    ManifoldFlowWrapper,
)
from gembed.core.module.bijection import (
    ContinuousAmbientFlow,
    RegularisedContinuousAmbientFlow,
    RegularisedContinuousManifoldFlow,
)
from gembed.nn.linear.concat_squash_linear import *
from gembed.core.module.spectral import FourierFeatureMap
from gembed.core.module.regression import (
    ResidualFCRegressionModule,
    ResidualDGCNRegressionModule,
)
from gembed.nn.residual import ResidualCoefficient


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
        t_dim=32,
        out_channels=3,
    ):
        super().__init__()

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        print(f"Fourier feature scale: {fourier_feature_scale}")
        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(in_channels, hidden_dim, fourier_feature_scale)
            self.ffm_t = nn.Linear(1, t_dim, fourier_feature_scale)
        else:
            self.ffm_x = FourierFeatureMap(
                in_channels, hidden_dim, fourier_feature_scale
            )
            self.ffm_t = FourierFeatureMap(1, t_dim, fourier_feature_scale)

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, c, t",
                    [
                        (nn.LayerNorm(hidden_dim), "x -> x"),  # ADDED v147
                        (
                            HyperConcatSquash(
                                hidden_dim,
                                hidden_dim,
                                n_context,
                                t_dim,
                                hyper_hidden_dim,
                            ),
                            "x, c, t -> x",
                        ),
                        nn.LayerNorm(hidden_dim),
                        nn.Softplus(),
                        (
                            HyperConcatSquash(
                                hidden_dim,
                                hidden_dim,
                                n_context,
                                t_dim,
                                hyper_hidden_dim,
                            ),
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
                (nn.LayerNorm(hidden_dim), "x -> x"),  # ADDED v147
                (
                    HyperConcatSquash(
                        hidden_dim, hidden_dim, n_context, t_dim, hyper_hidden_dim
                    ),
                    "x,c,t -> x",
                ),
                nn.Softplus(),
                (
                    HyperConcatSquash(
                        hidden_dim, out_channels, n_context, t_dim, hyper_hidden_dim
                    ),
                    "x,c,t -> x",
                ),
                ResidualCoefficient(),
            ],
        )

    def forward(self, t, x, c, **kwargs):
        # prep input
        x = self.ffm_x(x)
        t = self.ffm_t(t.unsqueeze(0))[None]

        for f in self.layers:
            x = x + f(x, c, t)

        # return velocity
        return self.regression(x, c, t)


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn):
        super().__init__()
        self.feature_nn = feature_nn

    def inverse(self, pos, batch):
        return self.feature_nn(pos, batch)

    def __str__(self):
        return str(self.__class__.str())


class RegularisedPointManifoldFlowSTNModel(RegularisedPointManifoldFlowSTN):
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
        stn = SpatialTransformer(
            ResidualDGCNRegressionModule(
                n_components=2 * 3,
                fourier_feature_scale=-1,
                n_hidden_layers=1,
                hidden_dim=64,
                dropout=False,
            )
        )

        # shape distribution model
        # sdm = None
        sdm = ShapeModel(
            ResidualFCRegressionModule(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale,
                n_hidden_layers=1,
                hidden_dim=64,
                dropout=False,
            )
        )

        # point distribution model
        AF_fdyn = FDyn(
            n_context=n_components,
            fourier_feature_scale=fourier_feature_scale,
            in_channels=2,
            out_channels=2,
            n_hidden_layers=1,
            hidden_dim=64,
            hyper_hidden_dim=64,
        )

        MF_fdyn = FDyn(
            n_context=n_components,
            fourier_feature_scale=fourier_feature_scale,
            in_channels=3,
            out_channels=3,
            n_hidden_layers=4,
            hidden_dim=128,
            hyper_hidden_dim=64,
        )

        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(
                torch.zeros(2),
                (1 / 3) * torch.eye(2, 2),
            ),
            layers=ManifoldFlowWrapper(
                ambient_flow=RegularisedContinuousAmbientFlow(
                    AF_fdyn,
                    noise_distribution=MultivariateNormal(
                        torch.zeros(2), torch.eye(2, 2)
                    ),
                    estimate_trace=True,
                    method=integration_method,
                    adjoint=adjoint,
                ),
                manifold_flow=RegularisedContinuousManifoldFlow(
                    MF_fdyn,
                    method=integration_method,
                    adjoint=adjoint,
                ),
            ),
        )

        super().__init__(sdm=sdm, pdm=pdm, stn=stn)
