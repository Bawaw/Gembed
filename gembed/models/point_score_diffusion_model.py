#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.mln import ContinuousDGM
from gembed.core.distribution import MultivariateNormal
from gembed.nn.fusion import HyperConcatSquash, LinearCombination, ConcatFuse
from copy import deepcopy
from gembed.core.module import (
    InvertibleModule,
    NormalisingFlow,
    # SingleLaneAugmentedPointFlow,
    # RegularisedSingleLaneAugmentedPointFlow,
    PointScoreDiffusionSTN,
)
from gembed.core.module.bijection import AbstractODE
from gembed.core.module.stochastic import VPSDE, SubVPSDE

from gembed.core.module.bijection import (
    ContinuousAmbientFlow,
    RegularisedContinuousAmbientFlow,
)
from gembed.nn.linear.concat_squash_linear import *
from gembed.core.module.spectral import FourierFeatureMap
from gembed.core.module.regression import (
    ResidualRegressionModule,
    ResidualDGCNRegressionModule,
)
from gembed.nn.residual import ResidualCoefficient
from scipy import integrate

from gembed.core.module.point_score_diffusion import Phase


class ConcatSquash(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        t_dim,
        hidden_dim,
        hyper_bias=None,
        hyper_gate=None,
        net=None,
    ):
        super().__init__()

        self.in_dim, self.out_dim, self.t_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            t_dim,
            hidden_dim,
        )
        if hyper_bias is None:
            # self.hyper_bias = nn.Sequential(
            #     nn.Linear(self.t_dim, self.out_dim),
            # )
            self.hyper_bias = nn.Sequential(
                # nn.LayerNorm(self.t_dim),
                nn.Linear(self.t_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_bias = hyper_bias

        if hyper_gate is None:
            # self.hyper_gate = nn.Sequential(
            #     nn.Linear(self.t_dim, self.out_dim),
            # )
            self.hyper_gate = nn.Sequential(
                # nn.LayerNorm(self.t_dim),
                nn.Linear(self.t_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_gate = hyper_gate

        if net is None:
            self.net = nn.Linear(self.in_dim, self.out_dim, bias=False)
        else:
            self.net = net

    def forward(self, x, t, batch):
        n_batch = batch.max() + 1

        # get weight matrices
        a = torch.sigmoid(self.hyper_gate(t.view(n_batch, self.t_dim)))
        b = self.hyper_bias(t.view(n_batch, self.t_dim))

        # evaluate layer: (a * (x @ W.T)) + b
        result = a * self.net(x) + b
        return result


class FCResidual(nn.Module):
    r""" Models the latent dynamics."""

    def __init__(
        self,
        hidden_dim=512,
        n_hidden_layers=5,
    ):
        super().__init__()

        # hidden layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    ResidualCoefficient(),
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        for f in self.layers:
            x = x + f(x)

        return self.regression(x)


class LatentFDyn(nn.Module):
    r""" Models the latent dynamics."""

    def __init__(
        self,
        in_channels,
        out_channels,
        fourier_feature_scale_t,
        n_hidden_layers=3,
        hyper_hidden_dim=512,
        hidden_dim=512,
        t_dim=32,
        layer_type="concatsquash",
        activation_type="swish",
    ):
        super().__init__()

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        self.ffm_x = nn.Linear(in_channels, hidden_dim)

        if fourier_feature_scale_t is None:
            self.ffm_t = nn.Linear(1, 32)
        else:
            self.ffm_t = FourierFeatureMap(1, 32, fourier_feature_scale_t)

        def layer(in_channels, out_channels):
            if layer_type == "linear_combination":
                return LinearCombination(in_channels, 32, out_channels)
            elif layer_type == "concatsquash":
                return ConcatSquash(
                    in_channels,
                    out_channels,
                    t_dim,
                    hyper_hidden_dim,
                )

        def activation():
            if activation_type == "softplus":
                return nn.Softplus()
            elif activation_type == "swish":
                return nn.SiLU()

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, t, batch",
                    [
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
                        # nn.LayerNorm(hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                        activation(),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
                        # (nn.LayerNorm(hidden_dim), "x -> x"),
                        ResidualCoefficient(),
                    ],
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, t, batch",
            [
                # # L1
                (
                    layer(hidden_dim, hidden_dim),
                    "x,t,batch -> x",
                ),
                nn.LayerNorm(hidden_dim),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L2
                (
                    layer(hidden_dim, hidden_dim),
                    "x,t,batch -> x",
                ),
                (nn.LayerNorm(hidden_dim), "x -> x"),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L3
                (
                    layer(hidden_dim, out_channels),
                    "x,t,batch -> x",
                ),
                ResidualCoefficient(),
            ],
        )

    def forward(self, x, t, batch, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if t.dim() == 0:
            t = t.repeat(batch.max() + 1, 1)

        # prep input
        x, t = self.ffm_x(x), self.ffm_t(t)

        for f in self.layers:
            x = x + f(x, t, batch=batch)

        # return velocity
        return self.regression(x, t, batch=batch)


class FDyn(nn.Module):
    r""" Models the dynamics."""

    def __init__(
        self,
        n_context,
        fourier_feature_scale_x,
        fourier_feature_scale_t,
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
        print(f"FFS_x: {fourier_feature_scale_x}, FFS_t: {fourier_feature_scale_t}")
        if fourier_feature_scale_x is None:
            self.ffm_x = nn.Linear(in_channels, hidden_dim)
        else:
            self.ffm_x = FourierFeatureMap(
                in_channels, hidden_dim, fourier_feature_scale_x
            )

        if fourier_feature_scale_t is None:
            self.ffm_t = nn.Linear(1, 32)
        else:
            self.ffm_t = FourierFeatureMap(1, 32, fourier_feature_scale_t)

        layer_type = "hyperconcatsquash"
        # layer_type = "linear_combination"
        # activation_type = "softplus"
        # activation_type = "tanh"
        activation_type = "swish"

        def layer(in_channels, out_channels):
            if layer_type == "hyperconcatsquash":
                return HyperConcatSquash(
                    in_channels,
                    out_channels,
                    n_context,
                    t_dim,
                    hyper_hidden_dim,
                )
            elif layer_type == "linear_combination":
                return LinearCombination(in_channels, n_context, t_dim, out_channels)

            elif layer_type == "concat_fuse":
                return ConcatFuse(in_channels, n_context, t_dim, out_channels)
            else:
                assert False

        def activation():
            if activation_type == "tanh":
                return nn.Tanh()
            elif activation_type == "tanhshrink":
                return nn.Tanhshrink()
            elif activation_type == "softplus":
                return nn.Softplus()
            elif activation_type == "swish":
                return nn.SiLU()
            else:
                assert False

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, c, t, batch",
                    [
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, c, t, batch -> x",
                        ),
                        # nn.LayerNorm(hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                        activation(),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, c, t, batch -> x",
                        ),
                        # (nn.LayerNorm(hidden_dim), "x -> x"),
                        ResidualCoefficient(),
                    ],
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, c, t, batch",
            [
                # # L1
                (
                    layer(hidden_dim, hidden_dim),
                    "x,c,t,batch -> x",
                ),
                nn.LayerNorm(hidden_dim),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L2
                (
                    layer(hidden_dim, hidden_dim),
                    "x,c,t,batch -> x",
                ),
                (nn.LayerNorm(hidden_dim), "x -> x"),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L3
                (
                    layer(hidden_dim, out_channels),
                    "x,c,t,batch -> x",
                ),
                ResidualCoefficient(),
            ],
        )

    def forward(self, x, t, c, batch, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        # prep input
        x, t = self.ffm_x(x), self.ffm_t(t)

        for f in self.layers:
            x = x + f(x, c, t, batch=batch)

        # return velocity
        return self.regression(x, c, t, batch=batch)


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn, add_log_var_module=False):
        super().__init__()
        self.feature_nn = feature_nn

        if add_log_var_module:
            # add log var regression module for VAEs
            self.log_var_regression = deepcopy(self.feature_nn.regression)

    def get_params(self, pos, batch):
        embedding = self.feature_nn.feature_forward(pos, batch)

        mean = self.feature_nn.regression(embedding)
        log_var = self.log_var_regression(embedding)

        return mean, log_var

    def inverse(self, pos, batch, return_params=False):
        return self.feature_nn(pos, batch)

    def __str__(self):
        return str(self.__class__.str())


class MetricTransformer(InvertibleModule):
    def __init__(self, feature_nn):
        super().__init__()
        # self.feature_nn = feature_nn

        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )
        self.encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

        #self._constant = nn.Parameter(torch.randn([1])[0])

    @property
    def constant(self):
        return F.relu(self._constant)

    def forward(self, Z, **kwargs):
        # return self.feature_nn.forward(Z, batch=torch.arange(Z.shape[0]), **kwargs)
        return self.decoder(Z)

    def inverse(self, Z, **kwargs):
        # return self.feature_nn.inverse(Z, batch=torch.arange(Z.shape[0]), **kwargs)
        return self.encoder(Z)

    def __str__(self):
        return str(self.__class__.str())


class PointScoreDiffusionSTNModel(PointScoreDiffusionSTN):
    def __init__(
        self,
        n_components=128,
        k=20,
        lambda_kld=1e-8,
        aggr="max",
        integration_method="rk4",
        fourier_feature_scale_x=1.0,
        fourier_feature_scale_t=1.0,
        beta_max=3,
        beta_min=1e-4,
        ltn_n_hidden_layers=5,
        ltn_hidden_dim=512,
        ltn_hyper_hidden_dim=128,
        sdm_n_hidden_layers=3,
        sdm_hidden_dim=128,
        pdm_n_hidden_layers=10,
        pdm_hidden_dim=128,
        pdm_hyper_hidden_dim=128,
        adjoint=False,
        use_stn=True,
        use_ltn=True,
        use_mtn=True,
    ):
        # NETWORK CONFIG

        # spatial transformer
        if use_stn:
            stn = SpatialTransformer(
                ResidualRegressionModule(
                    n_components=2 * 3,
                    fourier_feature_scale=-1,
                    n_hidden_layers=2,
                    hidden_dim=64,
                    layer_type="pointnet",
                )
            )

        else:
            stn = None

        if use_ltn:
            ltn = VPSDE(
                # pdm = SubVPSDE(
                beta_min=beta_min,
                beta_max=8,
                f_score=LatentFDyn(
                    in_channels=n_components,
                    out_channels=n_components,
                    fourier_feature_scale_t=fourier_feature_scale_t,
                    n_hidden_layers=ltn_n_hidden_layers,
                    hidden_dim=ltn_hidden_dim,
                    hyper_hidden_dim=ltn_hyper_hidden_dim,
                ),
            )
        else:
            ltn = None

        if use_mtn:
            mtn = MetricTransformer(
                feature_nn=AbstractODE(
                    dynamics=LatentFDyn(
                        in_channels=512,
                        out_channels=512,
                        fourier_feature_scale_t=30,
                        n_hidden_layers=0,
                        hidden_dim=512,
                        hyper_hidden_dim=128,
                        layer_type="linear_combination",
                        activation_type="softplus",
                    ),
                    adjoint=False,
                    method="rk4",
                )
            )
        else:
            mtn = None

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=ResidualRegressionModule(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale_x,
                n_hidden_layers=sdm_n_hidden_layers,
                hidden_dim=sdm_hidden_dim,
                layer_type="pointnet",
            ),
            add_log_var_module=lambda_kld > 0,
        )

        # point distribution model
        pdm = VPSDE(
            # pdm = SubVPSDE(
            beta_min=beta_min,
            beta_max=beta_max,
            f_score=FDyn(
                n_context=n_components,
                fourier_feature_scale_x=fourier_feature_scale_x,
                fourier_feature_scale_t=fourier_feature_scale_t,
                n_hidden_layers=pdm_n_hidden_layers,
                hidden_dim=pdm_hidden_dim,
                hyper_hidden_dim=pdm_hyper_hidden_dim,
                in_channels=3,
                out_channels=3,
            ),
        )

        super().__init__(
            sdm=sdm, pdm=pdm, stn=stn, ltn=ltn, mtn=mtn, lambda_kld=lambda_kld
        )


def get_mtn():
    print("TODO: remove this")
    mtn = MetricTransformer(
        feature_nn=AbstractODE(
            dynamics=LatentFDyn(
                in_channels=512,
                out_channels=512,
                fourier_feature_scale_t=30,
                n_hidden_layers=0,
                hidden_dim=512,
                hyper_hidden_dim=128,
                layer_type="linear_combination",
                activation_type="softplus",
            ),
            adjoint=False,
            method="rk4",
        )
    )

    return mtn
