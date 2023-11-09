# #!/usr/bin/env python3

# #TODO: turn into more flexible module

# class TransformerBlock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_dim=512):
#         super().__init__()
#         # self.lin_in = nn.Sequential(
#         #     nn.Linear(in_channels, in_channels),
#         # )
#         # self.lin_out = nn.Sequential(
#         #     # nn.Linear(out_channels, out_channels),
#         #     nn.LayerNorm(),
#         #     nn.ELU(),
#         # )
#         # self.lin_in = nn.Linear(in_channels, in_channels)
#         # self.lin_out = nn.Linear(out_channels, out_channels)

#         # self.pos_nn = nn.Sequential(
#         #     nn.Linear(3, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ELU(),
#         #     nn.Linear(hidden_dim, out_channels),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ELU(),
#         # )

#         # self.attn_nn = nn.Sequential(
#         #     nn.Linear(out_channels, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ELU(),
#         #     nn.Linear(hidden_dim, out_channels),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.ELU(),
#         # )
#         self.pos_nn = None
#         self.attn_nn = None

#         self.transformer = PointTransformerConv(
#             in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
#         )

#     def forward(self, x, pos, edge_index):
#         # x = self.lin_in(x)
#         x = self.transformer(x, pos, edge_index)
#         # x = self.lin_out(x)
#         return x


# class TransitionDown(torch.nn.Module):
#     """
#     Samples the input point cloud by a ratio percentage to reduce
#     cardinality and uses an mlp to augment features dimensionnality
#     """

#     def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
#         super().__init__()
#         self.k = k
#         self.ratio = ratio
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#         )

#     def forward(self, x, pos, batch):
#         # FPS sampling
#         id_clusters = fps(pos, ratio=self.ratio, batch=batch)

#         # compute for each cluster the k nearest points
#         sub_batch = batch[id_clusters] if batch is not None else None

#         # beware of self loop
#         id_k_neighbor = knn(
#             pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch
#         )

#         # transformation of features through a simple MLP
#         x = self.mlp(x)

#         # Max pool onto each cluster the features from knn in points
#         x_out = scatter_max(
#             x[id_k_neighbor[1]],
#             id_k_neighbor[0],
#             dim=0,
#             dim_size=id_clusters.size(0),
#         )[0]

#         # keep only the clusters and their max-pooled features
#         sub_pos, out = pos[id_clusters], x_out
#         return out, sub_pos, sub_batch


# class TransformerRegressionModule(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, fourier_feature_scale, dim_model, k=20
#     ):
#         super().__init__()
#         self.k = k

#         if fourier_feature_scale == -1:
#             self.ffm_x = nn.Linear(in_channels, dim_model[0])
#         else:
#             self.ffm_x = FourierFeatureMap(
#                 in_channels, dim_model[0], fourier_feature_scale
#             )

#         # FEATURE LEARNING
#         self.transformer_input = TransformerBlock(
#             in_channels=dim_model[0], out_channels=dim_model[0]
#         )

#         self.transformers_down = torch.nn.ModuleList()
#         self.transition_down = torch.nn.ModuleList()

#         for i in range(len(dim_model) - 1):
#             # Add Transition Down block followed by a Transformer block
#             self.transition_down.append(
#                 TransitionDown(
#                     in_channels=dim_model[i], out_channels=dim_model[i + 1], k=self.k
#                 )
#             )

#             self.transformers_down.append(
#                 TransformerBlock(
#                     in_channels=dim_model[i + 1], out_channels=dim_model[i + 1]
#                 )
#             )

#         # self.mlp_output = nn.Sequential(
#         #     nn.Linear(dim_model[-1], 64),
#         #     nn.ELU(),
#         #     nn.Linear(64, out_channels),
#         # )
#         self.mlp_output = nn.Sequential(
#             # L1
#             nn.Linear(dim_model[-1], dim_model[-1]),
#             nn.LayerNorm(dim_model[-1]),
#             nn.ELU(),
#             # L1
#             nn.Linear(dim_model[-1], dim_model[-1]),
#             nn.LayerNorm(dim_model[-1]),
#             nn.ELU(),
#             # L3
#             nn.Linear(dim_model[-1], out_channels),
#         )

#     def forward(self, pos, batch):
#         x = self.ffm_x(pos)
#         edge_index = knn_graph(pos, k=self.k, batch=batch)
#         x = self.transformer_input(x, pos, edge_index)

#         for i in range(len(self.transformers_down)):
#             x, pos, batch = self.transition_down[i](x, pos, batch=batch)

#             edge_index = knn_graph(pos, k=self.k, batch=batch)
#             x = self.transformers_down[i](x, pos, edge_index)

#         x = global_max_pool(x, batch)

#         out = self.mlp_output(x)
#         # return condition
#         return out

#     def __str__(self):
#         return str(self.__class__.str())
