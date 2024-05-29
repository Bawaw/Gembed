#!/usr/bin/env python3

import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from gembed.stats import frechet_mean
from torch_scatter import scatter_mean
from gembed.stats.geodesic import discrete_geodesic
from helper import load_experiment, pathcat, pyvista_save_kwargs, pyvista_plot_kwargs, get_plot_cdim

def plot_results(Xs, X_hat_dict, file_path=None):
    if file_path is None:
        PYVISTA_PLOT_KWARGS = pyvista_plot_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        Xs = [(X.cpu(), None) for X in Xs]

        for k in X_hat_dict.keys():
            Xs.append((X_hat_dict[k].cpu(), None))

        plot_objects(*Xs, **PYVISTA_PLOT_KWARGS)
    else:
        PYVISTA_SAVE_KWARGS = pyvista_save_kwargs(EXPERIMENT_NAME)
        cdim = get_plot_cdim(EXPERIMENT_NAME)

        os.makedirs(file_path, exist_ok=True)
        for i, X in enumerate(Xs):
            plot_objects((X.cpu(), None), snapshot_file_name=pathcat(file_path, f"X_{i}.png"), **PYVISTA_SAVE_KWARGS)

        for k in X_hat_dict.keys():
            plot_objects((X_hat_dict[k].cpu(), None), snapshot_file_name=pathcat(file_path, f"{k}.png"), **PYVISTA_SAVE_KWARGS)

def linear_point_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps, device):
    Xs = torch.stack([model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1] for Z in Zs.unsqueeze(1)])

    # compute mean
    X_hat = frechet_mean(Xs, n_iters=0)

    # return mean
    return X_hat[0]

def linear_latent_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps, device):
    # compute mean
    Z_hat = frechet_mean(Zs, n_iters=0)

    # Synthesise mean shape
    _, X_hat = model.forward(Z_hat, apply_mtn=False, apply_pdm=True)
    X_hat = f_refine(X_hat, Z_hat, n_refinement_steps)

    # return mean
    return X_hat

def linear_metric_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps, device):
    # embed in metric space
    Rs = torch.stack([model.mtn.inverse(Z) for Z in Zs])

    # compute mean
    R_hat = frechet_mean(Rs, n_iters=0)

    # Synthesise mean shape
    Z_hat, X_hat = model.forward(R_hat, apply_mtn=True, apply_pdm=True)
    X_hat = f_refine(X_hat, Z_hat, n_refinement_steps)

    # return mean
    return X_hat

def riemannian_latent_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps, device,
                                  n_random_point_samples=10, n_frechet_iters=1000, n_geodesic_iters=1000, decimals=4, n_geodesic_cps=6):
    # local metric is euclidean distance
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

    # global metric is geodesic energy
    global_metric = lambda Z0, Z1: discrete_geodesic(
        Z0, Z1, f_local_metric=MSEMetric(n_random_point_samples, n_geodesic_cps, Z0.device),
        verbose=True,
        return_energy=True,
        n_iters=100,
        n_cps=6,
        decimals=3,
    )[1]

    # compute mean
    Z_hat = frechet_mean(Zs, n_iters=10, f_global_metric=global_metric, verbose=True)

    # Synthesise mean shape
    _, X_hat = model.forward(Z_hat, apply_mtn=False, apply_pdm=True)
    X_hat = f_refine(X_hat, Z_hat, n_refinement_steps)

    # return mean
    return X_hat


def run(model, T_sample, f_refine, dataset, n_random_point_samples=8192, device="cpu", file_path=None, n_refinement_steps=6):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()

    # normalise dataa
    dataset = [T_norm(d) for d in dataset]
    file_path = pathcat(file_path, "-".join([str(d.id) for d in dataset]))

    # extract embeddings
    Zs = torch.concat(
        [
            model.inverse(
                T_sample(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True, apply_mtn=False
            )
            for d in dataset
        ]
    )

    # setup template
    z_template = 0.8 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )

    # compute means
    X_hats = {}
    X_hats["X_hat_linear_point_space"] = linear_point_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps=n_refinement_steps, device=device)
    X_hats["X_hat_linear_latent_space"] = linear_latent_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps=n_refinement_steps, device=device)
    X_hats["X_hat_linear_metric_space"] = linear_metric_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps=n_refinement_steps, device=device)
    X_hats["X_hat_riemannian_latent_space"] = riemannian_latent_space_atlas(model, T_sample, f_refine, Zs, z_template, n_refinement_steps=n_refinement_steps, device=device)

    # plot_results
    plot_results(dataset, X_hats, file_path)


def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    run(model, T_sample, f_refine, train[:3], device=device, file_path=file_path)

if __name__ == "__main__":
    main()