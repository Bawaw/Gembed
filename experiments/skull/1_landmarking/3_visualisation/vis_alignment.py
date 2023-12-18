#!/usr/bin/env python3

import lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.vis import Plotter
from gembed.core.optim import gradient_ascent
from transform import SubsetSample
from gembed.transforms import RandomRotation, RandomTranslation, Clip


def plot_shapes(Xs1, Xs2):
    plotter = Plotter(shape=(1, 2))
    for X1, X2 in zip(Xs1, Xs2):
        plotter.subplot(0, 0)
        plotter.add_generic(X1, color="#699d94")

        plotter.subplot(0, 1)
        plotter.add_generic(X2, color="#699d94")

    plotter.link_views()

    plotter.show()


def vis_alignment(
    model,
    dataset,
    device="cpu",
):
    pl.seed_everything(42, workers=True)

    # shape sampler
    f_sample_points = lambda n_samples: (
        tgt.SamplePoints(n_samples)
        if hasattr(dataset[0], "face")
        else SubsetSample(n_samples)
    )

    # normalisation and autmentation transform
    T_normalise = tgt.NormalizeScale()

    T = tgt.Compose(
        [
            f_sample_points(8192),
            tgt.NormalizeScale(),
        ]
    )

    model = model.to(device)

    norm_dataset = [T_normalise(d.clone()) for d in dataset]

    # augment the data and align the same data using the STN
    aligned_dataset = []
    for data in dataset:
        data_aligned = data.clone()

        # align the data
        _, params = model.stn(
            T(data.clone()).pos.to(device),
            batch=None,
            return_params=True,
        )
        data_aligned.pos = model.stn(
            T_normalise(data.clone()).pos.to(device), batch=None, params=params
        )

        aligned_dataset.append(data_aligned.to("cpu"))

    plot_shapes(norm_dataset, aligned_dataset)
