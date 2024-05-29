#!/usr/bin/env python3

import os
import torch
import pyvista as pv
import seaborn as sns
import lightning as pl
import matplotlib.pyplot as plt
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
from gembed.utils.adapter import torch_geomtric_data_to_vtk
from helper import load_experiment, pathcat, pyvista_save_kwargs


def plot_px_grids(log_px_grids, file_path):
    for t, log_px_grid in enumerate(log_px_grids):
        ax = sns.heatmap(log_px_grid, xticklabels=False, yticklabels=False)
        ax.invert_yaxis()

        if file_path is None:
            plt.show()
        else:
            os.makedirs(file_path, exist_ok=True)
            ax.get_figure().savefig(pathcat(file_path, f"X_{t}.png"), bbox_inches="tight", dpi=300)

        plt.close()

def latent_space_sm(model, Zs, x_grid, n_components, time_steps, plot_log_px, grid_size, device, file_path):
    # fit PCA model to the space
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(Zs.cpu())

    # setup density evaluation plane
    ## iterate PCs
    for pc in range(n_components):
        Rs_pc = torch.zeros(time_steps, n_components)
        Rs_pc[:, pc] = torch.linspace(-3, 3, time_steps) 
        Zs_sample = torch.from_numpy(pca.inverse_transform(Rs_pc)).float()

        px_grids = []
        # iterate time_steps
        for i, Z_sample in enumerate(Zs_sample.unsqueeze(1)):
            log_px_grid = model.log_prob(
                x_grid.clone(), batch=None, condition=Z_sample.to(device)
            )

            if not plot_log_px:
                log_px_grid = torch.exp(log_px_grid)
            px_grids.append(log_px_grid.view(grid_size, grid_size).cpu())

        _file_path = pathcat(file_path, f"pc_{pc}")
        os.makedirs(_file_path, exist_ok=True)
        plot_px_grids(px_grids, file_path=_file_path)

def run(
        model, 
        T_sample, 
        dataset, 
        n_components,
        depth=0.0,
        time_steps=9, 
        grid_size = 128, 
        plane="xy",
        plot_log_px=False,
        device="cpu", 
        file_path=None, 
):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()

    # extract embeddings
    Xs = [T_norm(d) for d in dataset]

    if model.stn is not None:
        stn_params = model.stn.get_transform_params(Xs[0].clone().pos.to(device), None)
    else:
        stn_params = None

    Zs = torch.concat([
        model.inverse(
            T_sample(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True, apply_mtn=False
        )
        for d in Xs
    ])

    # create pixel density grid
    x_grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)],
            indexing="xy",
        ),
        -1,
    )

    (x1, x2), x_3 = x_grid.view(-1, 2).T, depth * torch.ones(grid_size ** 2)

    if plane == "xy":
        # convert pixel grid to point cloud
        x_grid = torch.stack([x1, x2, x_3], -1).to(device)
    else:
        raise NotImplementedError()

    latent_space_sm(model, Zs, x_grid, n_components, time_steps, plot_log_px, grid_size, device, file_path)


def main():
    import sys

    (
        experiment_name, model, T_sample, _, _, train, _, _, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, train, n_components=10, device=device, file_path=pathcat(file_path, "train_px"))

if __name__ == "__main__":
    main()