#!/usr/bin/env python3

import os

from sklearn import mixture

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.core.optim import gradient_ascent
from gembed.vis import plot_objects
from transform import SubsetSample


def sample_random_shape(
    model,
    dataset,
    n_random_shape_samples,
    n_random_point_samples=10000,
    n_refinement_steps=0,
    device="cpu",
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
    print("TODO: change this to base distribution samples")
    Zs = torch.randn(n_random_shape_samples, 512)

    for i, Z in enumerate(Zs):
        print(f"Condition vector: {Z}")
        Z = Z[None].to(device)
        C = model.ltn.forward(Z)
        X = model.forward(C, apply_pdm=True, apply_ltn=False)[1]

        if n_refinement_steps > 0:
            print("Starting refinement...")
            X = gradient_ascent(
                init_x=X.requires_grad_(True),
                f_grad=lambda x, b, c: model.pdm.score(
                    x, torch.Tensor([0.0]).to(x.device), b, c
                ),
                condition=C,
                batch_size=3000,  # 7000,
                n_steps=n_refinement_steps,
            ).detach()

        X = X.cpu()

        if snapshot_root is not None:
            os.makedirs(snapshot_root, exist_ok=True)
            save_file_path = os.path.join(snapshot_root, f"random_shape_{i}.png")
        else:
            save_file_path = None

        plot_objects(
            (X, None),
            snapshot_file_name=save_file_path,
            color="#cccccc",
            cmap="cool",
        )
