#!/usr/bin/env python3
import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from torch_scatter import scatter_mean
from gembed.vis import plot_features_2D
from gembed.stats.geodesic import discrete_geodesic
from helper import load_experiment, pathcat, PYVISTA_SAVE_KWARGS, PYVISTA_PLOT_KWARGS

def plot_interpolation_trajectory(Z_trajectories, Zs, file_path):
    # plot embeddings
    fig = plot_features_2D(
        Zs,
        file_name=pathcat(file_path, "interpolation_trajectories.svg"),
        umap=False,
        Z_trajs=Z_trajectories,
    )


def plot_shapes(Xs, file_path):
    if file_path is None:
        plot_objects(*[(X.cpu(), None) for X in Xs], **PYVISTA_PLOT_KWARGS)

    else:
        os.makedirs(file_path, exist_ok=True)
        for i, X in enumerate(Xs):
           plot_objects((X.cpu(), None), snapshot_file_name=pathcat(file_path, f"X_{i}.png"), **PYVISTA_SAVE_KWARGS)

def linear_point_space_interpolation(model, f_refine, Z0, Z1, z_template, n_geodesic_cps, file_path, n_refinement_steps):
    # synthesise source
    X0 = model.forward(Z0, z=z_template, apply_ltn=False, apply_pdm=True)[1][None]
    X0 = f_refine(X0, Z0, n_refinement_steps)

    # synthesise target
    X1 = model.forward(Z1, z=z_template, apply_ltn=False, apply_pdm=True)[1][None]
    X1 = f_refine(X1, Z1, n_refinement_steps)

    # interpolate between source and target
    Xs = discrete_geodesic(
        X0, X1,
        f_local_metric=lambda x, y: (x - y).pow(2).sum(-1).mean(-1),
        n_iters=0,
        n_cps=n_geodesic_cps,
    )

    plot_shapes(Xs, file_path)

def linear_latent_space_interpolation(model, f_refine, Z0, Z1, z_template, n_geodesic_cps, file_path, n_refinement_steps):
    # linear interpolate between representations in latent space
    Zs = discrete_geodesic(Z0, Z1, n_iters=0, n_cps=n_geodesic_cps)

    # convert representations to shapes and refine
    Xs = torch.stack([
        f_refine(model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
        for Z in Zs.unsqueeze(1)
    ])

    # plot the shapes
    plot_shapes(Xs, file_path)

    return Zs.cpu()


def linear_metric_space_interpolation(model, f_refine, Z0, Z1, z_template, n_geodesic_cps, file_path, n_refinement_steps, keep_start_end=True):
    # project shapes to metric space
    Z0_metric, Z1_metric = model.mtn.inverse(Z0), model.mtn.inverse(Z1)

    # linear inteprolation in metric space
    Zs_metric = discrete_geodesic(
        Z0_metric, Z1_metric,
        f_local_metric=lambda x, y: (x - y).pow(2).sum(-1).mean(-1),
        n_iters=0,
        n_cps=n_geodesic_cps,
    )

    # project points back to latent space
    Zs = model.mtn.forward(Zs_metric)

    # overwrite start and finish of curve (to prevent reconstruction error)
    if keep_start_end:
        Zs[0], Zs[-1] = Z0, Z1

    # convert representations to shapes and refine
    Xs = torch.stack([
        f_refine(model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
        for Z in Zs.unsqueeze(1)
    ])

    # plot the shapes
    plot_shapes(Xs, file_path)

    return Zs.cpu()

def riemannian_latent_space_interpolation(model, f_refine, Z0, Z1, z_template, n_geodesic_cps, file_path, n_refinement_steps, n_random_point_samples=8192, n_iters=1000, decimals=4):
    class MSEMetric:
        def __init__(self, n_samples, n_geodesic_cps, device):
            # Construct a (typically smaller), template  that is used in optimisation
            self.z_optim_template = (
                torch.randn(n_samples, 3).repeat(n_geodesic_cps - 1, 1).to(device)
            )
            self.batch_optim_template = (
                torch.concat(
                    [i * torch.ones(n_samples) for i in range(n_geodesic_cps - 1)]
                )
                .long()
                .to(device)
            )

        def __call__(self, Zs_i, Zs_j):
            # generate sources
            _, Xs_i = model.forward(
                Z=Zs_i,
                z=self.z_optim_template,
                batch=self.batch_optim_template,
                apply_pdm=True,
                apply_ltn=False,
                time_steps=10,
            )

            # generated targets
            _, Xs_j = model.forward(
                Z=Zs_j,
                z=self.z_optim_template,
                batch=self.batch_optim_template,
                apply_pdm=True,
                apply_ltn=False,
                time_steps=10,
            )

            # MSE between sources and targets in shape space
            return scatter_mean((Xs_i - Xs_j).pow(2).sum(-1), self.batch_optim_template)

    # optimise geodesic in latent space using the above metric
    Zs = discrete_geodesic(
        Z0,
        Z1,
        f_local_metric=MSEMetric(n_random_point_samples, n_geodesic_cps, Z0.device),
        verbose=True,
        n_iters=n_iters,
        n_cps=n_geodesic_cps,
        decimals=decimals,
    )

    # convert representations to shapes and refine
    Xs = torch.stack([
        f_refine(model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
        for Z in Zs.unsqueeze(1)
    ])

    # plot the shapes
    plot_shapes(Xs, file_path)

    return Zs.cpu()

def main(
    model,
    T_sample,
    f_refine,
    dataset,
    data_0,
    data_1,
    n_refinement_steps=6,
    n_random_point_samples=8192,
    device="cpu",
    file_path=None,
    n_geodesic_cps=6,
    riemannian_kwargs={},
):
    pl.seed_everything(42, workers=True)

    T = tgt.Compose(
        [
            T_sample,
            tgt.NormalizeScale(),
        ]
    )

    # setup template
    X_template = T(data_0.clone()).pos
    z_template = 0.8 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )

    # setup source and target
    X0 = T(data_0.clone()).pos.to(device)
    X1 = T(data_1.clone()).pos.to(device)

    Z0, params0 = model.inverse(
        X0, None, apply_stn=True, return_params=True, apply_ltn=False
    )
    Z1, params1 = model.inverse(
        X1, None, apply_stn=True, return_params=True, apply_ltn=False
    )

    # Compute embeddings
    Zs = torch.concat(
        [
            model.inverse(
                T(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True
            )
            for d in dataset
        ]
    ).cpu()

    # interpolate
    Zs_trajs = {}

    linear_point_space_interpolation(
        model, f_refine, Z0, Z1, z_template, n_geodesic_cps, pathcat(file_path, "linear_point_space_interpolation"), n_refinement_steps
    )

    Zs_trajs["Linear Interpolation"] = linear_latent_space_interpolation(
        model, f_refine, Z0, Z1, z_template, n_geodesic_cps, pathcat(file_path, "linear_latent_space_interpolation"), n_refinement_steps
    )

    Zs_trajs["Linear MS Interpolation"] = linear_metric_space_interpolation(
        model, f_refine, Z0, Z1, z_template, n_geodesic_cps, pathcat(file_path, "linear_metric_space_interpolation"), n_refinement_steps
    )

    Zs_trajs[
        "Riemannian Interpolation"
    ] = riemannian_latent_space_interpolation(
        model, f_refine, Z0, Z1, z_template, n_geodesic_cps, pathcat(file_path, "riemannian_latent_space_interpolation"), n_refinement_steps, **riemannian_kwargs
    )

    # plot latent interpolation trajectories
    plot_interpolation_trajectory(Zs_trajs, Zs, file_path)


if __name__ == "__main__":
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        X0, X1 = train[0].clone(), train[7].clone()
        main(model, T_sample, f_refine, train + valid + test, X0, X1, device=device, file_path=pathcat(file_path, f"train/{X0.id}-{X1.id}"))
