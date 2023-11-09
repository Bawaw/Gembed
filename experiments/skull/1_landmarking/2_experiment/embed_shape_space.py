#!/usr/bin/env python3

import os


import pytorch_lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.core.optim import gradient_ascent
from gembed.vis import plot_features_2D
from transform import SubsetSample


def embed_shape_space(
    model,
    dataset,
    device="cpu",
    umap_metric="euclidean",
    snapshot_root=None,
):
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(dataset[0], "face") else SubsetSample(8192)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )

    # setup model
    model = model.to(device)

    # Compute embeddings
    Zs = torch.concat(
        [
            model.inverse(
                T(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True
            )
            for d in dataset
        ]
    ).cpu()

    # import matplotlib.pyplot as plt

    # plt.scatter(Zs[:, 0], Zs[:, 1])
    # plt.show()

    # if snapshot_root is not None:
    #     os.makedirs(snapshot_root, exist_ok=True)
    #     save_file_path = os.path.join(snapshot_root, f"shape_space.svg")
    # else:
    #     save_file_path = None

    # # plot embeddings
    # fig = plot_features_2D(
    #     Zs,
    #     file_name=save_file_path,
    #     umap_kwargs={"metric": "euclidean", "min_dist": 0.9, "spread": 0.9},
    # )

    return Zs
