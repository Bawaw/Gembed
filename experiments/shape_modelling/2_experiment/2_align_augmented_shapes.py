#!/usr/bin/env python3

import os
import torch
import pandas as pd
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import Plotter
from helper import load_experiment, pathcat, PYVISTA_SAVE_KWARGS, PYVISTA_PLOT_KWARGS
from gembed.utils.transforms import RandomRotation, RandomTranslation

def _plot(data_set, file_name):
    os.makedirs(file_path, exist_ok=True)

    plotter = Plotter()
    for data in data_set:
        plotter.add_generic(d, **PYVISTA_save_KWARGS)

    plotter.screenshot(file_name, transparent_background=True)
    plotter.close()

def plot_result(results, file_path):
    if file_path is None:
        for identifier, augmented_and_aligned_data in results:
            plotter = Plotter(shape=(1, 2))

            for d, d_aligned in shape_data:
                plotter.subplot(0, 0)
                plotter.add_generic(d, **PYVISTA_PLOT_KWARGS)

                plotter.subplot(0, 1)
                plotter.add_generic(d_aligned, **PYVISTA_PLOT_KWARGS)

            plotter.link_views()
            plotter.show()
    else:
        for identifier, augmented_and_aligned_data in results:
            augmented_data, algined_data = zip(*results)
            _plot(augmented_data, pathcat(file_path, f"{identifier}/augmented/"))
            _plot(aligned_data, pathcat(file_path, f"{identifier}/aligned"))



def quantify_results(results, file_path):
    # function computes average MSE to the mean shape
    f_mse = lambda x: (x - x.mean(0)).pow(2).sum(-1).mean(-1)

    identifiers, augmented_data_mse, aligned_data_mse = [], [], []
    for identifier, augmented_and_aligned_data in results:

        identifiers.append(identifier)
        augmented_data_mse.append(mse(torch.stack([d_aug.pos for d_aug, _ in augmented_and_aligned_data])))
        aligned_data_mse.append(mse(torch.stack([d_align.pos for _, d_align in augmented_and_aligned_data])))

    df = pd.DataFrame({
        "id" : identifiers,
        "augmented_data_mse" : augmented_data_mse,
        "aligned_data_mse" : aligned_data_mse,
    })

    df.to_csv(file_path)

def main(
    model,
    T_sample,
    dataset,
    n_aug_samples=5,
    n_refinement_steps=6,
    device="cpu",
    file_path=None,
):
    pl.seed_everything(42, workers=True)

    # setup data transforms
    T_normalise = tgt.NormalizeScale()
    T_augment = tgt.Compose(
        [tgt.NormalizeScale(), RandomRotation(sigma=0.2), RandomTranslation(sigma=0.1)]
    )

    # augment the data and align using the STN
    results = []

    for data in dataset:
        augmented_and_aligned_data = []
        for _ in range(n_aug_samples):
            # augment data
            data_augmented = T_augment(T_normalise(data.clone()))

            # compute the alignment parameters
            _, params = model.stn(
                T_sample(data_augmented.clone()).pos.to(device),
                batch=None,
                return_params=True,
            )

            # apply alignment parameters to shape
            data_aligned = data.clone()
            data_aligned.pos = model.stn(
                data_augmented.clone().pos.to(device), batch=None, params=params
            )

            # add raw augmented
            augmented_and_aligned_data.append((data_augmented.to("cpu"), data_aligned.to("cpu")))

        results.append((data.id, augmented_and_aligned_data))

        # compute and store the average to the mean in in augmented and aligned data
        results.append((
            data.id,
            mse(torch.stack([d_aug.pos for d_aug, _ in augmented_and_aligned_data])),
            mse(torch.stack([d_align.pos for _, d_align in augmented_and_aligned_data]))
        ))

    plot_result(results)
    quantify_result(results)


if __name__ == "__main__":
    assert model.stn is not None, "Model does not have STN!"
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        main(model, T_sample, train[:5], device=device, file_path=file_path)
