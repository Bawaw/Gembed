#!/usr/bin/env python3

import os
import torch
import pyvista as pv
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from sklearn.decomposition import PCA, KernelPCA
from gembed.vis.plotter import Plotter
from helper import load_experiment, pathcat, pyvista_save_kwargs

def plot_video(Xs, z_scalars, file_name):
    # get kwargs
    PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)

    if "window_size" in PYVISTA_SAVE_KWARGS.keys():
        del PYVISTA_SAVE_KWARGS["window_size"]

    # loop frames
    Xs = Xs + Xs[::-1] # append revese

    # setup plotter
    plotter = Plotter(off_screen=True)

    plotter.open_gif(file_name)
    plotter.camera_position = PYVISTA_SAVE_KWARGS.pop("camera_position")

    # iterate frames
    for t, X in enumerate(Xs):
        # if first frame
        if t == 0:
            plotter.add_generic(
                X,
                render_points_as_spheres=True,
                scalars=z_scalars,
                **PYVISTA_SAVE_KWARGS
            )
            plotter.remove_scalar_bar()

        else:
            plotter.update_coordinates(X.detach().cpu())

            plotter.write_frame()

    plotter.close()

def plot_frames(Xs, x_scalars, file_path):
    PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)

    for t, X in enumerate(Xs):
        plot_objects((X, x_scalars), snapshot_file_name=pathcat(file_path, f"X_{t}.png"), **PYVISTA_SAVE_KWARGS)


def latent_space_sm(model, Zs, z_template, z_scalars, n_components, time_steps, f_refine, n_refinement_steps, device, file_path):
    # fit PCA model to the space
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(Zs.cpu())

    # Synthesise shape variation
    ## iterate PCs
    for pc in range(n_components):
        Rs_pc = torch.zeros(time_steps, n_components)
        Rs_pc[:, pc] = torch.linspace(-3, 3, time_steps) 
        Zs_sample = torch.from_numpy(pca.inverse_transform(Rs_pc)).float()

        Xs = []
        # iterate time_steps
        for i, Z_sample in enumerate(Zs_sample.unsqueeze(1)):
            Z_hat, X_hat = model.forward(Z_sample.to(device), z=z_template, apply_mtn=False, apply_pdm=True)
            Xs.append(f_refine(X_hat, Z_hat, n_refinement_steps).cpu())

        # Save results for 1 PC
        _file_path = pathcat(file_path, f"pc_{pc}")
        os.makedirs(_file_path, exist_ok=True)
        plot_frames(Xs, z_scalars, _file_path)
        plot_video(Xs, z_scalars, pathcat(_file_path, f"pc_{pc}.gif"))

def run(model, T_sample, f_refine, dataset, n_components, time_steps=9, n_random_point_samples=int(8e5), n_refinement_steps=6, device="cpu", file_path=None):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()

    # extract embeddings
    Xs = [T_norm(d) for d in dataset]

    Zs = torch.concat([
        model.inverse(
            T_sample(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True, apply_mtn=False
        )
        for d in Xs
    ])

    # Synthesise shape variation
    z_template = 0.8 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )

    z_scalars = z_template[:, 0].cpu()
    
    latent_space_sm(model, Zs, z_template, z_scalars, n_components, time_steps, f_refine, n_refinement_steps, device, file_path)


def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, _, train, _, _, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, f_refine, train, n_components=10, device=device, file_path=pathcat(file_path, "train"))

if __name__ == "__main__":
    main()