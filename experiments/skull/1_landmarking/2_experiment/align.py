#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.vis import Plotter
from gembed.core.optim import gradient_ascent
from transform import SubsetSample
from gembed.transforms import RandomRotation, RandomTranslation, Clip


def align_augmented_data(
    model,
    dataset,
    device="cpu",
    n_iterations=5,
    snapshot_root=None,
):
    pl.seed_everything(42, workers=True)

    # average MSE to the mean shape
    mse = lambda x: (x - x.mean(0)).pow(2).sum(-1).mean(-1)

    # shape sampler
    f_sample_points = lambda n_samples: (
        tgt.SamplePoints(n_samples)
        if hasattr(dataset[0], "face")
        else SubsetSample(n_samples)
    )

    # normalisation and autmentation transform
    T_normalise = tgt.NormalizeScale()
    T_augment = tgt.Compose(
        [tgt.NormalizeScale(), RandomRotation(sigma=0.2), RandomTranslation(sigma=0.1)]
    )

    T_sample = f_sample_points(8192)

    model = model.to(device)

    # augment the data and align the same data using the STN
    aligned_dataset = []
    for data in dataset:
        aligned_data = []
        for _ in range(n_iterations):
            # transform data
            data_aligned = data.clone()
            data_augmented = T_augment(T_normalise(data.clone()))

            # align the data
            _, params = model.stn(
                T_sample(data_augmented.clone()).pos.to(device),
                batch=None,
                return_params=True,
            )
            data_aligned.pos = model.stn(
                data_augmented.clone().pos.to(device), batch=None, params=params
            )

            aligned_data.append((data_augmented.to("cpu"), data_aligned.to("cpu")))

        mse_augmented = mse(torch.stack([d_aug.pos for d_aug, _ in aligned_data]))
        mse_aligned = mse(torch.stack([d_align.pos for _, d_align in aligned_data]))

        aligned_dataset.append((aligned_data, mse_augmented, mse_aligned))

    ##### Plot the aligned data #####
    for aligned_data in aligned_dataset:
        data, mse_augmented, mse_aligned = aligned_data

        plotter = Plotter(shape=(1, 2))
        for d, d_aligned in data:
            plotter.subplot(0, 0)
            plotter.add_generic(d)

            plotter.subplot(0, 1)
            plotter.add_generic(d_aligned)

        print(
            f"MSE augmented: {mse_augmented.mean()}, MSE aligned: {mse_aligned.mean()}"
        )
        plotter.link_views()
        plotter.show()
    #################################
