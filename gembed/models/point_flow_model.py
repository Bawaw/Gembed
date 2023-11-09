#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.mln import ContinuousDGM
from gembed.core.distribution import MultivariateNormal
from gembed.nn.fusion import HyperConcatSquash, ConcatFuse, LinearCombination
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

        self.ffm_x = FourierFeatureMap(
            3,
            2048,
            1.0,
        )
        self.ffm_t = FourierFeatureMap(1, 32, 30)

        # ENCODER
        self.e1 = tgnn.Sequential(
            "x, t",
            [
                (LinearCombination(2048, 32, 1024), "x, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )
        self.e2 = tgnn.Sequential(
            "x, t",
            [
                (LinearCombination(1024, 32, 512), "x, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(512),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )
        self.e3 = tgnn.Sequential(
            "x, t",
            [
                (LinearCombination(512, 32, 256),"x, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(256),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )

        self.e4 = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        # DECODER
        self.d4 = torch.nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )


        self.d3 = tgnn.Sequential(
            "x, x2, t",
            [
                (LinearCombination(256, 256, 32, 512),"x, x2, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(512),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )
        self.d2 = tgnn.Sequential(
            "x, x2, t",
            [
                (LinearCombination(512, 512, 32, 1024),"x, x2, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(1024),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )
        self.d1 = tgnn.Sequential(
            "x, x2, t",
            [
                (LinearCombination(1024, 1024, 32, 2048),"x, x2, t -> x"),
                (torch.nn.Sequential(
                    nn.LayerNorm(2048),
                    nn.SiLU(),
                ), "x -> x")
            ],
        )

        self.d0 = torch.nn.Sequential(nn.Linear(2048, 3), ResidualCoefficient())

    def forward(self, t, x, c, **kwargs):
        #original_t = t.unsqueeze(1)
        t = t[None, None]

        # prep input
        x = self.ffm_x(x)
        t = self.ffm_t(t)

        # encoder
        i1 = self.e1(x, t)
        i2 = self.e2(i1, t)
        i3 = self.e3(i2, t)
        i4 = self.e4(i3)

        # decoder
        o4 = self.d4(i4)
        o3 = self.d3(o4, i3, t)
        o2 = self.d2(o3, i2, t)
        o1 = self.d1(o2, i1, t)

        x = self.d0(o1)

        return x


# class FDyn(nn.Module):
#     r""" Models the dynamics."""

#     def __init__(
#         self,
#         n_context,
#         fourier_feature_scale,
#         in_channels=3,
#         n_hidden_layers=3,
#         hyper_hidden_dim=512,
#         hidden_dim=512,
#         t_dim=32,
#         out_channels=3,
#     ):
#         super().__init__()

#         # small scale = large kernel (underfitting)
#         # large scale = small kernel (overfitting)
#         print(f"Fourier feature scale: {fourier_feature_scale}")
#         if fourier_feature_scale == -1:
#             self.ffm_x = nn.Linear(in_channels, hidden_dim, fourier_feature_scale)
#             self.ffm_t = nn.Linear(1, t_dim, fourier_feature_scale)
#         else:
#             # self.ffm_x = FourierFeatureMap(
#             #     in_channels, hidden_dim, fourier_feature_scale
#             # )
#             # self.ffm_t = FourierFeatureMap(1, t_dim, fourier_feature_scale)
#             self.ffm_x = FourierFeatureMap(
#                 in_channels, hidden_dim, fourier_feature_scale
#             )
#             self.ffm_t = FourierFeatureMap(1, t_dim, 30)

#         layer_type = "hyperconcatsquash"
#         # layer_type = "linear_combination"
#         activation_type = "softplus"

#         def layer(in_channels, out_channels):
#             if layer_type == "hyperconcatsquash":
#                 return HyperConcatSquash(
#                     in_channels,
#                     out_channels,
#                     n_context,
#                     t_dim,
#                     hyper_hidden_dim,
#                 )
#             elif layer_type == "linear_combination":
#                 return LinearCombination(
#                     True, in_channels, n_context, t_dim, out_channels
#                 )

#             elif layer_type == "concat_fuse":
#                 return ConcatFuse(in_channels, n_context, t_dim, out_channels)
#             else:
#                 assert False

#         def activation():
#             if activation_type == "tanh":
#                 return nn.Tanh()
#             elif activation_type == "tanhshrink":
#                 return nn.Tanhshrink()
#             elif activation_type == "softplus":
#                 return nn.Softplus()
#             else:
#                 assert False

#         # hidden layers
#         self.layers = nn.ModuleList(
#             [
#                 tgnn.Sequential(
#                     "x, c, t",
#                     [
#                         (nn.LayerNorm(hidden_dim), "x -> x"),
#                         (
#                             layer(hidden_dim, hidden_dim),
#                             # HyperConcatSquash(
#                             #     hidden_dim,
#                             #     hidden_dim,
#                             #     n_context,
#                             #     t_dim,
#                             #     hyper_hidden_dim,
#                             # ),
#                             "x, c, t -> x",
#                         ),
#                         nn.LayerNorm(hidden_dim),
#                         activation(),
#                         (
#                             layer(hidden_dim, hidden_dim),
#                             # HyperConcatSquash(
#                             #     hidden_dim,
#                             #     hidden_dim,
#                             #     n_context,
#                             #     t_dim,
#                             #     hyper_hidden_dim,
#                             # ),
#                             "x, c, t -> x",
#                         ),
#                         ResidualCoefficient(),
#                     ],
#                 )
#                 for _ in range(n_hidden_layers)
#             ]
#         )

#         # final regression layer
#         self.regression = tgnn.Sequential(
#             "x, c, t",
#             [
#                 (nn.LayerNorm(hidden_dim), "x -> x"),
#                 (
#                     layer(hidden_dim, hidden_dim),
#                     # HyperConcatSquash(
#                     #     hidden_dim, hidden_dim, n_context, t_dim, hyper_hidden_dim
#                     # ),
#                     "x,c,t -> x",
#                 ),
#                 activation(),
#                 (
#                     layer(hidden_dim, out_channels),
#                     # HyperConcatSquash(
#                     #     hidden_dim, 3, n_context, t_dim, hyper_hidden_dim
#                     # ),
#                     "x,c,t -> x",
#                 ),
#                 ResidualCoefficient(),
#             ],
#         )

#     def forward(self, t, x, c, **kwargs):
#         # prep input
#         x = self.ffm_x(x)
#         t = self.ffm_t(t.unsqueeze(0))[None]
#         # t = t.repeat(x.shape[0], 1)

#         for f in self.layers:
#             x = x + f(x, c, t)

#         # return velocity
#         return self.regression(x, c, t)


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn):
        super().__init__()
        self.feature_nn = feature_nn
        self.z = torch.from_numpy(np.random.RandomState(42).randn(1, 64)).float()

    def inverse(self, pos, batch):
        #return self.feature_nn(pos, batch)
        return self.z.to(pos.device)

    def __str__(self):
        return str(self.__class__.str())


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
        stn = None
        # stn = SpatialTransformer(
        #     ResidualDGCNRegressionModule(
        #         n_components=2 * 3,
        #         fourier_feature_scale=-1,
        #         n_hidden_layers=1,
        #         hidden_dim=64,
        #         dropout=False,
        #     )
        # )

        # shape distribution model
        # sdm = None
        sdm = ShapeModel(
            # ResidualFCRegressionModule(
            #     n_components=n_components,
            #     fourier_feature_scale=fourier_feature_scale,
            #     n_hidden_layers=1,
            #     hidden_dim=64,
            #     dropout=False,
            # )
            ResidualDGCNRegressionModule(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale,
                n_hidden_layers=2,
                hidden_dim=128,
                dropout=False,
            )
        )

        # point distribution model
        fdyn = FDyn(
            n_context=n_components,
            fourier_feature_scale=fourier_feature_scale,
            n_hidden_layers=1,
            hidden_dim=128,
            # hidden_dim=64,
            hyper_hidden_dim=128,  # ,128,
            # hyper_hidden_dim=256,
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


def point_flow_model(pretrained=False, progress=True, **kwargs):
    model = PointFlowModel(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
    #     model.load_state_dict(state_dict)
    return model
