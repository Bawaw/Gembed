#!/usr/bin/env python3

import torch
import torch.nn as nn


class SpatialTransformer(nn.Module):
    def __init__(self, feature_nn, regressor_nn, zero_initialise=True):
        super().__init__()

        # feature learning
        self.feature = feature_nn

        # y = Ax + b
        self.fc = regressor_nn

        # set weights and bias of last layer to 0
        # if zero_initialise:
        #     self.fc[-1].weight.data.zero_()
        #     self.fc[-1].bias.data.copy_(
        #         torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float)
        #     )

    def rotation_activation(self, theta, pos, batch):
        # θ ∈ [-π, π]

        (
            alpha,
            beta,
            gamma,
        ) = theta.T

        # rotation matrix
        # source: https://wikimedia.org/api/rest_v1/media/math/render/svg/234c5831df9d48e5dc4a1cc130475d3426a64ce1
        R = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(beta) * torch.cos(gamma),
                        torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma)
                        - torch.cos(alpha) * torch.sin(gamma),
                        torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)
                        + torch.sin(alpha) * torch.sin(gamma),
                    ],
                    -1,
                ),
                torch.stack(
                    [
                        torch.cos(beta) * torch.sin(gamma),
                        torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma)
                        + torch.cos(alpha) * torch.cos(gamma),
                        torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)
                        - torch.sin(alpha) * torch.cos(gamma),
                    ],
                    -1,
                ),
                torch.stack(
                    [
                        -torch.sin(beta),
                        torch.sin(alpha) * torch.cos(beta),
                        torch.cos(alpha) * torch.cos(beta),
                    ],
                    -1,
                ),
            ],
            -1,
        ).to(pos.device)

        return R

    def get_transform_params(self, pos, batch):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

        # get features
        X = self.feature(pos, batch)

        # get transformations
        return self.fc(X)

    def forward(self, pos, batch, params=None):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

        # compute transformation arguments based on input
        if params is None:
            params = self.get_transform_params(pos, batch)

        # get transformation arguments
        theta, beta = params[:, :3], params[:, 3:]

        # TRANSLATE
        pos = pos + beta[batch]

        # ROTATE
        # [N x 3 x 3] x [N x 3 x 1] = [N x 3 x 1]
        rotation_matrices = self.rotation_activation(theta, pos, batch)
        pos = torch.bmm(pos[:, None, :], rotation_matrices[batch]).squeeze(1)

        return pos

    def inverse(self, pos, batch, params=None):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

        # compute transformation arguments based on input
        if params is None:
            params = self.get_transform_params(pos, batch)

        # get transformation arguments
        theta, beta = params[:, :3], params[:, 3:]

        # Inverse ROTATE
        # [N x 3 x 3] x [N x 3 x 1] = [N x 3 x 1]
        # The inverse of a rotation matrix is its transpose, which is also a rotation matrix:
        # https://en.wikipedia.org/wiki/Rotation_matrix#Multiplication
        rotation_matrices = self.rotation_activation(theta, pos, batch).mT
        pos = torch.bmm(pos[:, None, :], rotation_matrices[batch]).squeeze(1)

        # Inverse TRANSLATE
        pos = pos - beta[batch]

        return pos
