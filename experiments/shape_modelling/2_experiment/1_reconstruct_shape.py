#!/usr/bin/env python3

import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from helper import load_experiment, pathcat, PYVISTA_SAVE_KWARGS, PYVISTA_PLOT_KWARGS

def plot_result(X, X_aligned, X_rec_sampled, X_rec_refined, X_rec, file_path=None):

    if file_path is None:
        plot_objects(
            (X.cpu(), None),
            (X_aligned.cpu(), None),
            (X_rec_sampled.cpu(), None),
            (X_rec_refined.cpu(), None),
            (X_rec.cpu(), None),
            **PYVISTA_PLOT_KWARGS
        )
    else:
        os.makedirs(file_path, exist_ok=True)
        plot_objects((X.cpu(), None), snapshot_file_name=pathcat(file_path, "X.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_aligned.cpu(), None), snapshot_file_name=pathcat(file_path, "X_aligned.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec_sampled.cpu(), None), snapshot_file_name=pathcat(file_path, "X_rec_sampled.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec_refined.cpu(), None), snapshot_file_name=pathcat(file_path, "X_rec_refined.png"), **PYVISTA_SAVE_KWARGS)
        plot_objects((X_rec.cpu(), None), snapshot_file_name=pathcat(file_path, "X_rec.png"), **PYVISTA_SAVE_KWARGS)


def main(
    model,
    T_sample,
    f_refine,
    dataset,
    n_random_point_samples=8192,
    time_steps=100,
    n_refinement_steps=6,
    device="cpu",
    file_path=None
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
            X_aligned = model.stn.forward(X, None, stn_params)
        else:
            X_aligned = X

        # 3) reconstruct the shape
        X_rec_sampled = model.pdm_forward(
            z=0.8*model.pdm.base_distribution.sample(int(8e4)).to(device),
            condition=Z,
            return_time_steps=False,
        )

        # 4) refine result
        X_rec_refined = f_refine(X_rec_sampled, Z, n_refinement_steps)

        # 5) invert stn
        if model.stn is not None:
            X_rec = model.stn.inverse(X_rec_refined, None, params)
        else:
            X_rec = X_rec_refined

        plot_result(X, X_aligned, X_rec_sampled, X_rec_refined, X_rec, file_path=pathcat(file_path, str(data.id)))


if __name__ == "__main__":
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        main(model, T_sample, f_refine, train[:5], device=device, file_path=pathcat(file_path, "train"))
        main(model, T_sample, f_refine, valid[:5], device=device, file_path=pathcat(file_path, "valid"))
        main(model, T_sample, f_refine, test[:5], device=device, file_path=pathcat(file_path, "test"))
