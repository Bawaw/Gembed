#!/usr/bin/env python3
import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from helper import load_experiment, pathcat, pyvista_plot_kwargs, pyvista_save_kwargs, get_plot_cdim

def plot_result(X, file_path):

    X = X.cpu()

    if file_path is None:
        PYVISTA_PLOT_KWARGS = pyvista_plot_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        plot_objects((X, X[:, cdim]), **PYVISTA_PLOT_KWARGS)
    else:
        PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot_objects((X, X[:, cdim]), snapshot_file_name=file_path, **PYVISTA_SAVE_KWARGS)

def run(
    model,
    f_refine,
    n_random_shape_samples=5,
    n_random_point_samples=int(8e5),
    n_refinement_steps=20,
    device="cpu",
    file_path=None,
):
    pl.seed_everything(42, workers=True)
    has_ltn = model.ltn is not None

    # get seeds for shapes
    if has_ltn:
        Zs = model.ltn.base_distribution.sample(n_random_shape_samples).to(device)
    else:
        Zs = torch.randn(n_random_shape_samples, model.n_components).to(device)

    # get seeds for point configuration
    z = 0.8*model.pdm.base_distribution.sample(n_random_point_samples).to(device)

    for i, Z in enumerate(Zs.unsqueeze(1)):
        # synthesise new shape
        H, X = model.forward(Z=Z, z=z, apply_ltn=has_ltn, apply_pdm=True)

        # refine the new shape
        X = f_refine(X, H, n_refinement_steps)

        # plot the results
        plot_result(X, pathcat(file_path, f"X_{i}.png"))



def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, f_refine, device=device, file_path=file_path)

if __name__ == "__main__":
    main()