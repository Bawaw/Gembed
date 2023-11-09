#!/usr/bin/env python3

import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects

from transform import SubsetSample

def model_likelihood(
    model,
    dataset,
    n_samples=10,
    n_point_samples=80000,
    device="cpu",
    sampled_vis_mesh=False,
    n_input_samples=4096,
):
    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )

    # point embedding for template mesh
    template_z = torch.randn(n_point_samples, 3)
    # template_scalar = template_z[:, 0]

    model = model.to(device)
    template_z = template_z.to(device)

    for i in range(n_samples):
        # grab data sample
        X_data = tgt.Center()(dataset[i].clone())
        X_data = T(X_data.clone()).to(device)

        if self.stn is not None:
            X_data = self.stn(X_data, None)

        condition = self.sdm.inverse(X_data, None)
        log_px = self.pdm.log_prob(
            X_data, batch=None, condition=condition
        )

        # MLE
        # -1/N sum_i log px_i
        ll = scatter_mean(log_px, batch_augmented)

        print(ll)
