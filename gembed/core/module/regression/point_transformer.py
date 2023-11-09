#!/usr/bin/env python3

import torch
import torch.nn as nn
from gembed.core.module.spectral import FourierFeatureMap
import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    PointTransformerConv,
    fps,
    global_mean_pool,
    global_max_pool,
    knn,
    knn_graph,
)
from torch_scatter import scatter_max


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.lin_in = nn.Sequential(
            nn.Linear(in_channels, in_channels),
        )
        self.lin_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
        )

        self.pos_nn = nn.Sequential(
            # FourierFeatureMap(3, hidden_dim, 1.0),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        self.attn_nn = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """
    Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality
    """

    def __init__(self, in_channels, out_channels, ratio, k):
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
        # TODO: add relu as final activation
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
        self,
        fourier_feature_scale,
        n_components,
        dim_model,
        hidden_dim,
        in_channels=3,
        k=16,
        ratio=0.5,
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
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    ratio=ratio,
                    k=k,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
                )
            )

        self.global_pool = tgnn.Sequential(
            "x, batch",
            [
                (tgnn.global_mean_pool, "x, batch -> x"),
            ],
        )

        self.regression = nn.Sequential(
            # REGRESSION MODULE
            # L1
            nn.Linear(dim_model[-1], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # L2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # L3
            nn.Linear(hidden_dim, n_components),
        )

    def feature_forward(self, pos, batch):
        x = self.ffm_x(pos)
        # x = torch.ones((pos.shape[0], 32), device=pos.get_device())

        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        x = self.global_pool(x, batch)

        return x

    def forward(self, pos, batch):
        x = self.feature_forward(pos, batch)

        # return condition
        return self.regression(x)

    def __str__(self):
        return str(self.__class__.str())


class PointNet(nn.Module):
    def __init__(
        self,
        fourier_feature_scale,
        n_components,
        hidden_dim,
        in_channels=3,
    ):
        super().__init__()

        if fourier_feature_scale == -1:
            self.ffm_x = nn.Linear(in_channels, 128)
        else:
            self.ffm_x = FourierFeatureMap(in_channels, 128, fourier_feature_scale)

        # FEATURE LEARNING
        self.feature_nn = nn.Sequential(
            # REGRESSION MODULE
            # L1
            nn.Linear(128, 128),
            # nn.LayerNorm(128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # L2
            nn.Linear(128, 128),
            # nn.LayerNorm(128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # L3
            nn.Linear(128, 128),
        )

        self.global_pool = tgnn.Sequential(
            "x, batch",
            [
                (nn.Linear(128, 512), "x -> x"),
                (tgnn.global_max_pool, "x, batch -> x"),
            ],
        )

        self.regression = nn.Sequential(
            # REGRESSION MODULE
            # L1
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # L2
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # L3
            nn.Linear(hidden_dim, n_components),
        )

    def feature_forward(self, pos, batch):
        x = self.ffm_x(pos)

        x = self.feature_nn(x)

        x = self.global_pool(x, batch)

        return x

    def forward(self, pos, batch):
        x = self.feature_forward(pos, batch)

        # return condition
        return self.regression(x)

    def __str__(self):
        return str(self.__class__.str())
