#!/usr/bin/env python3

import torch
import torch.nn as nn


class LeastSquaresSpatialTransformer(nn.Module):
    """TODO: optimise and debug, it's not working"""

    def __init__(self, feature_nn, template, k=50):
        super().__init__()

        self.feature = feature_nn
        self.template = template
        self.k = k

    def forward(self, pos, batch):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

        n_batch = batch.max() + 1

        # template features
        batch_template = torch.zeros(self.template.shape[0]).to(pos.device)
        X_template = self.feature(self.template.to(pos.device), batch_template)

        # batch template features
        batch_template = torch.cat([batch_template + i for i in range(n_batch)])
        X_template = X_template.repeat(n_batch, 1)

        # object features
        X = self.feature(pos, batch)

        # find nearest neighbour in feature space
        _, knn_index = tgnn.knn(X_template, X, 1, batch_template, batch)

        # nn similarity in feature space
        S = 1 / (1 + (X_template[knn_index] - X).pow(2).sum(-1))

        # get indices, sorted per batch based on distance
        sorted_indices = topk(S, 1.0, batch)

        # take only the first k
        topk_indices = torch.stack(
            [sorted_indices[batch == b][: self.k] for b in range(n_batch)], 0
        )

        # setup moving and fixed PC
        M = pos.clone()  # .requires_grad_(True)
        F = self.template.repeat(n_batch, 1)[knn_index].to(
            pos.device
        )  # .requires_grad_(True).to(pos.device)[knn_index]

        # select the topk positions
        M = M[topk_indices]
        F = F[topk_indices]

        # one pad coordinates
        M = torch.cat([M, torch.ones(*M.shape[:2], 1).to(M.device)], -1)
        F = torch.cat([F, torch.ones(*F.shape[:2], 1).to(F.device)], -1)

        # A = argmax ||AM - F||
        A = torch.linalg.lstsq(M, F).solution

        # affine transform and exclude last dimension
        padded_pos = torch.cat([pos, torch.ones(pos.shape[0], 1).to(M.device)], -1)

        # [N x 1 x 4] = [N x 1 x 4] x [N x 4 x 4]
        # batched version of (pos @ A)
        ppos_transformed = torch.bmm(padded_pos[:, None, :], A[batch]).squeeze(1)

        return ppos_transformed[..., :-1]
