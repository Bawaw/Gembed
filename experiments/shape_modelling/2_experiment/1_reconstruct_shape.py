#!/usr/bin/env python3

import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from helper import load_experiment, pathcat, clean_result, pyvista_plot_kwargs, pyvista_save_kwargs, get_plot_cdim

def plot_result(X, X_aligned, X_rec_sampled, X_rec_refined, X_rec, file_path=None):

    X_rec = clean_result(X_rec.cpu())

    X, X_aligned, X_rec_sampled, X_rec_refined, X_rec = X.cpu(), X_aligned.cpu(), X_rec_sampled.cpu(), X_rec_refined.cpu(), X_rec.cpu()

    if file_path is None:
        PYVISTA_PLOT_KWARGS = pyvista_plot_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        plot_objects(
            (X, None),
            (X_aligned, X_aligned[:, cdim].clamp(-1, 1)),
            (X_rec_sampled, X_rec_sampled[:, cdim].clamp(-1, 1)),
            (X_rec_refined, X_rec_refined[:, cdim].clamp(-1, 1)),
            (X_rec.cpu(), X_rec[:, cdim].clamp(-1, 1)),
            **PYVISTA_PLOT_KWARGS,
        )
    else:
        PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)
        
        os.makedirs(file_path, exist_ok=True)
        plot_objects((X, None), snapshot_file_name=pathcat(file_path, "X.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_aligned, X_aligned[:, cdim].clamp(-1, 1)), snapshot_file_name=pathcat(file_path, "X_aligned.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec_sampled, X_rec_sampled[:, cdim].clamp(-1, 1)), snapshot_file_name=pathcat(file_path, "X_rec_sampled.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec_refined, X_rec_refined[:, cdim].clamp(-1, 1)), snapshot_file_name=pathcat(file_path, "X_rec_refined.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec, X_rec[:, cdim].clamp(-1, 1)), snapshot_file_name=pathcat(file_path, "X_rec.png"), **PYVISTA_SAVE_KWARGS)


def run(
    model,
    T_sample,
    f_refine,
    dataset,
    n_random_point_samples=int(8e5),
    n_refinement_steps=6,
    device="cpu",
    file_path=None,
):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()

    for i, data in enumerate(dataset):
        # 1) original data
        X = T_norm(data.clone())

        # get condition
        Z, stn_params = model.inverse(
            T_sample(X.clone()).pos.to(device),
            batch=None,
            apply_stn=True,
            return_params=True,
        )

        # 2) superimpose shape
        if model.stn is not None:
            X_aligned = model.stn.forward(X.pos.to(device), None, stn_params)
        else:
            X_aligned = X.pos

        # 3) reconstruct sampled shape 
        X_rec_sampled = model.pdm_forward(
            z=0.8*model.pdm.base_distribution.sample(n_random_point_samples).to(device),
            condition=Z,
            return_time_steps=False,
        )

        # 4) refine sampled shape
        X_rec_refined = f_refine(X_rec_sampled, Z, n_refinement_steps)

        # 5) Reconstruct meshed shape
        # grid_size = 256
        # grid = torch.stack(
        #     torch.meshgrid([ 
        #         #torch.Tensor([0.]),
        #         torch.linspace(-1.1, 1.1, grid_size), 
        #         torch.linspace(-1, 1, grid_size),
        #         torch.linspace(-1.1, 1.1, grid_size)
        #         ], indexing="xy",
        #     ), -1).view(-1, 3).to(device)

        # mag_grad_field = 1.*(model.pdm.score(
        #     grid, torch.Tensor([0.01]).to(grid), None, Z
        # ).pow(2).sum(-1).view(grid_size, grid_size,grid_size) < 1000)

        # import pyvista as pv
        # img_data = pv.ImageData(dimensions=(grid_size, grid_size, grid_size))
        # img_data.contour([1], mag_grad_field.cpu().numpy().ravel(order="F"), method="marching_cubes").plot(color="lightgray")

        # 5) invert stn
        if model.stn is not None:
            X_rec = model.stn.inverse(X_rec_refined, None, stn_params)
        else:
            X_rec = X_rec_refined

        plot_result(X, X_aligned, X_rec_sampled, X_rec_refined, X_rec, file_path=pathcat(file_path, str(data.id)))


def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, f_refine, train[:5], device=device, file_path=pathcat(file_path, "train"))
        run(model, T_sample, f_refine, valid[:5], device=device, file_path=pathcat(file_path, "valid"))
        run(model, T_sample, f_refine, test[:5], device=device, file_path=pathcat(file_path, "test"))

if __name__ == "__main__":
    main()
