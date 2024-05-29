#!/usr/bin/env python3

import os
import torch
import pandas as pd
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import Plotter
from helper import load_experiment, pathcat, pyvista_save_kwargs, pyvista_plot_kwargs, get_plot_cdim
from gembed.utils.transforms import RandomRotation, RandomTranslation

def _plot(data_samples, file_path):
    PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)
    PYVISTA_SAVE_KWARGS["cmap"] = "tab10"

    plotter = Plotter(off_screen=True) 
    plotter.camera_position = PYVISTA_SAVE_KWARGS.pop("camera_position")

    for i, d in enumerate(data_samples):
        plotter.add_generic(d, scalars=i*torch.ones(d.pos.shape[0]), show_scalar_bar = False, clim=[0, len(data_samples)], **PYVISTA_SAVE_KWARGS)

    plotter.screenshot(file_path, transparent_background=True)
    plotter.close()

def plot_result(results, file_path):
    if file_path is None:
        PYVISTA_PLOT_KWARGS = pyvista_plot_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        for identifier, augmented_and_aligned_data in results:
            plotter = Plotter(shape=(1, 2))
            plotter.camera_position = PYVISTA_PLOT_KWARGS.pop("camera_position")
            for i, (d, d_aligned) in enumerate(augmented_and_aligned_data):
                plotter.subplot(0, 0)
                plotter.add_generic(d, **PYVISTA_PLOT_KWARGS, scalars=i*torch.ones(d.pos.shape[0]))

                plotter.subplot(0, 1)
                plotter.add_generic(d_aligned, **PYVISTA_PLOT_KWARGS, scalars=i*torch.ones(d.pos.shape[0]))

            plotter.link_views()
            plotter.show()
    else:
        for identifier, augmented_and_aligned_data in results:
            augmented_data, aligned_data = zip(*augmented_and_aligned_data)
            
            _file_path = pathcat(file_path, f"{identifier}")
            os.makedirs(_file_path, exist_ok=True)
            _plot(augmented_data, pathcat(_file_path, "augmented"))
            _plot(aligned_data, pathcat(_file_path, "aligned"))



def quantify_results(results, file_path):
    # Σ_Nn [Σ_d(x_i - x_mean)^2] mean over points and number of augmentation samples
    f_mse = lambda x: (x - x.mean(0)).pow(2).sum(-1).mean().item()

    identifiers, augmented_data_mse, aligned_data_mse = [], [], []
    for identifier, augmented_and_aligned_data in results:
        identifiers.append(identifier)

        # compute and store the average to the mean shape in in augmented and aligned data
        augmented_data_mse.append(f_mse(torch.stack([d_aug.pos for d_aug, _ in augmented_and_aligned_data])))
        aligned_data_mse.append(f_mse(torch.stack([d_align.pos for _, d_align in augmented_and_aligned_data])))

    df = pd.DataFrame({
        "id" : identifiers,
        "augmented_data_mse" : augmented_data_mse,
        "aligned_data_mse" : aligned_data_mse,
    })

    df.to_csv(pathcat(file_path, "error_table.csv"))

def run(
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

            # [(aug_1, al_1), ...]
            augmented_and_aligned_data.append((data_augmented.to("cpu"), data_aligned.to("cpu")))

        # [id, [(aug_1, al_1), (aug_2, al_2), ...], ... ]
        results.append((data.id, augmented_and_aligned_data))

    plot_result(results, file_path)
    quantify_results(results, file_path)

def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    assert model.stn is not None, "Model does not have STN!"

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, test[:5], device=device, file_path=pathcat(file_path, "test"))

if __name__ == "__main__":
    main()