#!/usr/bin/env python3

import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from sklearn.decomposition import PCA, KernelPCA
from gembed.vis.plotter import Plotter
from helper import load_experiment, pathcat, PYVISTA_SAVE_KWARGS

def plot_result(Xs, z_scalars, file_name):
    # animate PC
    plotter = Plotter(off_screen=True)
    kwargs = PYVISTA_SAVE_KWARGS
    if "window_size" in kwargs.keys():
        del kwargs["window_size"]

    plotter.open_gif(file_name)
    plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]

    Xs = Xs + Xs[::-1] # append revese

    for t, X in enumerate(Xs):
        # if first frame
        if t == 0:
            plotter.add_generic(
                X.cpu(),
                render_points_as_spheres=True,
                scalars=z_scalars,
                **kwargs
            )
            plotter.remove_scalar_bar()

        else:
            plotter.update_coordinates(X.detach().cpu())

            plotter.write_frame()

    plotter.close()

def main(model, T_sample, f_refine, dataset, pc=1, n_random_point_samples=8192, time_steps=10, n_refinement_steps=6, device="cpu", file_path=None):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()

    # extract embeddings
    Xs = [T_norm(d) for d in dataset]

    Rs = torch.concat(
        [
            model.inverse(
                T_sample(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True, apply_mtn=True
            )
            for d in Xs
        ]
    )

    # Construct PCA-model
    pca = PCA(n_components=128, whiten=True)
    #pca = KernelPCA(n_components=128, fit_inverse_transform=True)
    pca.fit(Rs.cpu())

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.pyplot import plot
    # plot(np.arange(10), pca.explained_variance_ratio_.cumsum()[:10])
    # plt.show()

    # Synthesise shape variation
    z_template = 0.8 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )

    z_scalars = z_template[:, 0].cpu()

    Rs_pc = torch.zeros(time_steps, 128)
    Rs_pc[:, pc-1] = torch.linspace(-3, 3, time_steps) # only vary PC
    Rs_pc = torch.from_numpy(pca.inverse_transform(Rs_pc)).float()

    Xs = []
    for R in Rs_pc.unsqueeze(1).to(device):
        Z_hat, X_hat = model.forward(R, z=z_template, apply_mtn=True, apply_pdm=True)
        Xs.append(f_refine(X_hat, Z_hat, n_refinement_steps))

    os.makedirs(file_path, exist_ok=True)
    plot_result(Xs, z_scalars, pathcat(file_path, f"pc_{pc}.gif"))


if __name__ == "__main__":
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        for i in range(1, 11):
            main(model, T_sample, f_refine, train, pc=i, device=device, file_path=pathcat(file_path, "train"))
