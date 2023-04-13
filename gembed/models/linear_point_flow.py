#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
from gembed.core.distribution import MultivariateNormal
from gembed.core.module import InvertibleModule
from gembed.core.module.bijection import ContinuousAmbientFlow
from gembed.models import NormalisingFlow
from gembed.core.module import InvertibleModule
from gembed.models.point_flow import AugmentedPointFlow
from gembed.models.stn import SpatialTransformer
from gembed.nn.linear import ConcatSquashLinear


class FDyn(nn.Module):
    """ Models the dynamics of the injection to the manifold. """

    def __init__(self, n_context):
        super().__init__()
        # expected format: N x (C * L)
        # +1 for time
        self.csl1 = ConcatSquashLinear(3, 128, n_context)
        self.csl2 = ConcatSquashLinear(128, 256, n_context)
        self.csl4 = ConcatSquashLinear(256, 128, n_context)
        self.csl5 = ConcatSquashLinear(128, 3, n_context)

    def forward(self, t, x, c, **kwargs):
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

    def point_representation(self, pos, batch):
        raise NotImplementedError()

    def shape_representation(self, pos, batch):
        raise NotImplementedError()

    def __str__(self):
        return str(self.__class__.str())

    @staticmethod
    def str():
        raise NotImplementedError()


class LinearPointFlow(AugmentedPointFlow):
    def __init__(self, n_components=128, k=20, aggr="max"):
        # point distribution model
        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
            layers=ContinuousAmbientFlow(
                FDyn(n_context=n_components),
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=True,
                method="rk4",
                adjoint=False,
            ),
        )

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 64, n_components),
                    (tgnn.global_max_pool, "x, batch -> x"),
                ],
            )
        )

        # spatial transformer model
        stn = SpatialTransformer(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
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
                nn.ReLU(),
                nn.Linear(32, 2 * 3),
            ),
        )

        super().__init__(sdm, pdm, stn)


from gembed.models.point_flow import SingleLaneAugmentedPointFlow

class SingleLaneLPF(SingleLaneAugmentedPointFlow):
    def __init__(self, n_components=128, k=20, aggr="max"):
        # point distribution model
        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
            layers=ContinuousAmbientFlow(
                FDyn(n_context=n_components),
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=False,
                method="rk4",
                adjoint=False,
            ),
        )

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
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

                    # Aggregate features
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 128, 1024),
                    (tgnn.global_max_pool, "x, batch -> x"),

                    # MLP v11
                    nn.Linear(1024, 512),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, n_components),
                ],
            )
        )

        # spatial transformer model
        stn = None
        # stn = SpatialTransformer(
        #     feature_nn=tgnn.Sequential(
        #         "pos, batch",
        #         [
        #             (
        #                 tgnn.DynamicEdgeConv(
        #                     nn.Sequential(
        #                         nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
        #                     ),
        #                     k,
        #                     aggr,
        #                 ),
        #                 "pos, batch -> x1",
        #             ),
        #             (
        #                 tgnn.DynamicEdgeConv( nn.Sequential(
        #                         nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
        #                     ),
        #                     k,
        #                     aggr,
        #                 ),
        #                 "x1, batch -> x2",
        #             ),
        #             (
        #                 tgnn.DynamicEdgeConv(
        #                     nn.Sequential(
        #                         nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
        #                     ),
        #                     k,
        #                     aggr,
        #                 ),
        #                 "x2, batch -> x3",
        #             ),
        #             (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
        #             nn.Linear(3 * 64, 16),
        #             (tgnn.global_max_pool, "x, batch -> x"),
        #         ],
        #     ),
        #     regressor_nn=nn.Sequential(
        #         nn.Linear(16, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 2 * 3),
        #     ),
        # )

        super().__init__(sdm, pdm, stn)

# class SingleLaneLPF(SingleLaneAugmentedPointFlow):
#     def __init__(self, n_components=128, k=20, aggr="max"):
#         # point distribution model
#         pdm = NormalisingFlow(
#             base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#             layers=ContinuousAmbientFlow(
#                 FDyn(n_context=n_components),
#                 noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
#                 estimate_trace=False,
#                 method="rk4",
#                 adjoint=False,
#             ),
#         )

#         # shape distribution model
#         sdm = ShapeModel(
#             feature_nn=tgnn.Sequential(
#                 "pos, batch",
#                 [
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 # nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 3, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 3, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64),
#                                 # nn.Linear(2 * 3, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64),
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
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 64, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64),
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64),
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
#                     (
#                         tgnn.DynamicEdgeConv(
#                             nn.Sequential(
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64)
#                                 # nn.Linear(2 * 64, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ELU(), nn.Linear(64, 64),
#                                 # nn.Linear(2 * 64, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 64),
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
#                     #             (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     #             nn.Linear(3 * 64, 16),
#                     #             (tgnn.global_max_pool, "x, batch -> x"),
#                     # 2nd NONLINEAR FEATURE LEARNER
#                     (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     nn.Linear(3 * 128, 1024),
#                     # nn.Linear(3 * 64, 1024),
#                     (tgnn.global_max_pool, "x, batch -> x"),
#                     # MLP v10
#                     # nn.Dropout(0.1),
#                     # nn.Linear(1024, 512),
#                     # nn.ELU(),
#                     # nn.Dropout(0.2),
#                     # nn.Linear(512, 256),
#                     # nn.ELU(),
#                     # nn.Dropout(0.2),
#                     # nn.Linear(256, n_components),
#                     #
#                     # MLP v11
#                     nn.Linear(1024, 512),
#                     nn.ELU(),
#                     nn.Dropout(0.1),
#                     nn.Linear(512, 256),
#                     nn.ELU(),
#                     nn.Dropout(0.1),
#                     nn.Linear(256, n_components),
#                     # NONLINEAR FEATURE LEARNER
#                     # (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#                     # Representation learning
#                     # nn.Linear(3 * 64, 128),
#                     # nn.ELU(),
#                     # nn.Linear(128, 128),
#                     # nn.ELU(),
#                     # nn.Linear(128, 128),
#                     # (tgnn.global_max_pool, "x, batch -> x"),
#                     # # Manifold Learning
#                     # nn.Linear(128, 128),
#                     # nn.ELU(),
#                     # nn.Linear(128, 128),
#                     # nn.ELU(),
#                     # nn.Linear(128, n_components),
#                     # REDUCED NONLINEAR FEATURE LEARNER
#                     # nn.Linear(3 * 64, n_components),
#                     # nn.ELU(),
#                     # nn.Linear(n_components, n_components),
#                     # nn.ELU(),
#                     # nn.Linear(n_components, n_components),
#                     # (tgnn.global_max_pool, "x, batch -> x"),
#                     # nn.Linear(n_components, n_components),
#                     # nn.ELU(),
#                     # nn.Linear(n_components, n_components),
#                 ],
#             )
#         )

#         # spatial transformer model
#         stn = None
#         # stn = SpatialTransformer(
#         #     feature_nn=tgnn.Sequential(
#         #         "pos, batch",
#         #         [
#         #             (
#         #                 tgnn.DynamicEdgeConv(
#         #                     nn.Sequential(
#         #                         nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
#         #                     ),
#         #                     k,
#         #                     aggr,
#         #                 ),
#         #                 "pos, batch -> x1",
#         #             ),
#         #             (
#         #                 tgnn.DynamicEdgeConv( nn.Sequential(
#         #                         nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
#         #                     ),
#         #                     k,
#         #                     aggr,
#         #                 ),
#         #                 "x1, batch -> x2",
#         #             ),
#         #             (
#         #                 tgnn.DynamicEdgeConv(
#         #                     nn.Sequential(
#         #                         nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
#         #                     ),
#         #                     k,
#         #                     aggr,
#         #                 ),
#         #                 "x2, batch -> x3",
#         #             ),
#         #             (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
#         #             nn.Linear(3 * 64, 16),
#         #             (tgnn.global_max_pool, "x, batch -> x"),
#         #         ],
#         #     ),
#         #     regressor_nn=nn.Sequential(
#         #         nn.Linear(16, 16),
#         #         nn.ReLU(),
#         #         nn.Linear(16, 2 * 3),
#         #     ),
#         # )

#         super().__init__(sdm, pdm, stn)


from gembed.models.point_flow import AugmentedPointFlow_3


class LinearPointFlow_2(AugmentedPointFlow_3):
    def __init__(self, n_components=128, k=20, aggr="max"):
        # point distribution model
        pdm = NormalisingFlow(
            base_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
            layers=ContinuousAmbientFlow(
                FDyn(n_context=n_components),
                noise_distribution=MultivariateNormal(torch.zeros(3), torch.eye(3, 3)),
                estimate_trace=False,
                method="rk4",
                adjoint=False,
            ),
        )

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x2, batch -> x3",
                    ),
                    (lambda xs: torch.concat(xs, dim=1), "[x1, x2, x3] -> x"),
                    nn.Linear(3 * 64, n_components),
                    (tgnn.global_max_pool, "x, batch -> x"),
                ],
            )
        )

        # spatial transformer model
        stn = SpatialTransformer(
            feature_nn=tgnn.Sequential(
                "pos, batch",
                [
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 3, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "pos, batch -> x1",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
                            ),
                            k,
                            aggr,
                        ),
                        "x1, batch -> x2",
                    ),
                    (
                        tgnn.DynamicEdgeConv(
                            nn.Sequential(
                                nn.Linear(2 * 64, 64), nn.ELU(), nn.BatchNorm1d(64)
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
                nn.ReLU(),
                nn.Linear(32, 2 * 3),
            ),
        )

        super().__init__(sdm, pdm, stn)


def linear_point_flow(pretrained=False, progress=True, **kwargs):
    model = LinearPointFlow(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
    #     model.load_state_dict(state_dict)
    return model
