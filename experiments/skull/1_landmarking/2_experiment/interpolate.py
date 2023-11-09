#!/usr/bin/env python3

import pathlib

import pyvista as pv
import torch
import torch_geometric.transforms as tgt
from gembed.stats.geodesic import discrete_geodesic, continuous_geodesic
from gembed.vis.plotter import Plotter
from gembed.vis import plot_objects
from transform import SubsetSample
from torch_scatter import scatter_mean
import pytorch_lightning as pl
from gembed.vis import plot_features_2D


def plot_interpolation_trajectory(Z_trajectories, Zs, snapshot_root):
    if snapshot_root is not None:
        os.makedirs(snapshot_root, exist_ok=True)
        save_file_path = os.path.join(snapshot_root, f"shape_space.svg")
    else:
        save_file_path = None

    # plot embeddings
    fig = plot_features_2D(
        Zs,
        file_name=save_file_path,
        umap_kwargs={"metric": "euclidean", "min_dist": 0.9, "spread": 0.9},
        Z_trajs=Z_trajectories,
    )


def plot_shapes(Xs, snapshot_root):
    objects_and_scalars = [(sample.cpu(), Xs[0, :, 0].cpu()) for sample in Xs]

    # when saving, save the plots seperately
    if snapshot_root is not None:
        for i, sample_and_scalar in enumerate(objects_and_scalars):
            os.makedirs(snapshot_root, exist_ok=True)
            save_file_path = os.path.join(
                snapshot_root,
                f"interpolated_shape_t_{i/(len(objects_and_scalars) - 1)}.png",
            )

            plot_objects(
                sample_and_scalar, snapshot_file_name=save_file_name, cmap="cool"
            )

    else:
        plot_objects(
            *objects_and_scalars, cmap="cool",
            #camera_position=[(10, 0, 0), (0, 0, 0), (0, 0, 1)], # brain
            #camera_position=[(-8, -8, 0), (0, 0, 0), (0, 0, 1)], # hippocampus
            camera_position=[(-7, -7, 6), (0, 0, 0), (0, 0, 1)], # skull
        )

def refine(model, X, Z, n_refinement_steps):
    from gembed.core.optim import gradient_langevin_dynamics, gradient_ascent

    if n_refinement_steps > 0:
        X_refined = gradient_ascent(
            init_x=X.requires_grad_(True),
            f_grad=lambda x, b, c: model.pdm.score(
                x, torch.Tensor([0.0]).to(x.device), b, c
            ),
            condition=Z.clone(),
            batch_size=3000,
            n_steps=n_refinement_steps,
        ).detach()

    return X_refined

def linear_point_interpolation(
        model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps
):
    X0 = refine(model, model.forward(Z0, z=z_template, apply_ltn=False, apply_pdm=True)[1][None], Z0, n_refinement_steps)
    X1 = refine(model, model.forward(Z1, z=z_template, apply_ltn=False, apply_pdm=True)[1][None], Z1, n_refinement_steps)

    # linear interpolate between corresponding points in shape space
    Xs = discrete_geodesic(
        X0, X1,
        f_local_metric=lambda x, y: (x - y).pow(2).sum(-1).mean(-1),
        n_iters=0,
        n_cps=n_geodesic_cps,
    )

    plot_shapes(Xs, snapshot_root)


def linear_latent_interpolation(
        model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps
):
    # linear interpolate between representations in latent space
    Zs = discrete_geodesic(Z0, Z1, n_iters=0, n_cps=n_geodesic_cps)

    # convert representations to shapes
    Xs = torch.stack(
        [
            refine(model, model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
            for Z in Zs.unsqueeze(1)
        ]
    )

    # plot the shapes
    plot_shapes(Xs, snapshot_root)

    return Zs.cpu()


def linear_metric_space_interpolation(
        model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps, keep_start_end=True
):
    Z0_metric, Z1_metric = model.mtn.inverse(Z0), model.mtn.inverse(Z1)

    Zs_metric = torch.lerp(
        input=Z0_metric,
        end=Z1_metric,
        weight=torch.linspace(0, 1, n_geodesic_cps)[:, None].to(Z0.device),
    )

    Zs = model.mtn.forward(Zs_metric)

    if keep_start_end:
        Zs[0], Zs[-1] = Z0, Z1

    # convert representations to shapes
    Xs = torch.stack(
        [
            refine(model, model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
            for Z in Zs.unsqueeze(1)
        ]
    )

    # # plot the shapes
    plot_shapes(Xs, snapshot_root)

    return Zs.cpu()


def riemannian_latent_interpolation_discrete(
    model,
    Z0,
    Z1,
    z_template,
    n_geodesic_cps,
    snapshot_root,
    n_refinement_steps,
    n_random_point_samples=8000,
    n_iters=1000,
):
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

            # MSE between sources and targets
            return scatter_mean((Xs_i - Xs_j).pow(2).sum(-1), self.batch_optim_template)

    # compute geodesic using the above metric
    Zs = discrete_geodesic(
        Z0,
        Z1,
        f_local_metric=MSEMetric(n_random_point_samples, n_geodesic_cps, Z0.device),
        verbose=True,
        n_iters=n_iters,
        n_cps=n_geodesic_cps,
        decimals=4,
    )

    # convert representations to shapes
    Xs = torch.stack(
        [
            refine(model, model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1], Z, n_refinement_steps)
            for Z in Zs.unsqueeze(1)
        ]
    )

    # plot the shapes
    plot_shapes(Xs, snapshot_root)

    return Zs.cpu()


def batch_jvp(func, inputs, model, v=None, create_graph=True):
    # https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/utils.py
    # foo = func(inputs)
    # breakpoint()
    ## /TEST

    batch_size = inputs.size(0)
    z_dim = inputs.size(1)

    z = model.pdm.sample_base(batch_size * int(100))

    batch = torch.concat([i * torch.ones(100) for i in range(batch_size)]).long()

    func = lambda Z: model.forward(
        Z,
        z=z,
        batch=batch,
        apply_pdm=True,
        apply_ltn=False,
        time_steps=10,
        n_samples=100,
        # adjoint_params=(new_Z),
    )[1].view(Z.shape[0], -1)

    # import numpy as np

    # func = lambda Z: Z @ torch.eye(512, 512).to(Z.device)

    # func = lambda Z: Z @ torch.from_numpy(
    #     np.random.RandomState(42).randn(512, 3)
    # ).float().to(Z.device)

    # J = torch.autograd.functional.jacobian(func, inputs[0], strict=True)
    #
    from gembed.numerics.jacobian import batch_jacobian

    outputs = func(inputs)
    J = batch_jacobian(outputs, inputs)

    return J

    # foo = torch.autograd.functional.jvp(
    #     func, inputs, v=v, create_graph=create_graph, strict=True
    # )[1]
    # jac = (
    #     torch.autograd.functional.jvp(
    #         func, inputs, v=v, create_graph=create_graph, strict=True
    #     )[1]
    #     .view(batch_size, z_dim, -1)
    #     .permute(0, 2, 1)
    # )
    # return jac


def euclidean_metric_G(f, z, model, zdot=None, create_graph=False):
    J = batch_jvp(f, z, model, create_graph=create_graph)

    # return J.T @ J[None]
    return torch.einsum("nij,nik->njk", J, J)


# https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py
# if zdot is None:
#     J = batch_jvp(f, z, model, create_graph=create_graph)
#     return torch.einsum("nij,nik->njk", J, J)
# else:
#     Jv = torch.autograd.functional.jvp(f, z, v=zdot, create_graph=create_graph)[1]
#     return torch.einsum("nij,nij->n", Jv, Jv)


# def euclidean_metric_G(decode, z, eta=0.2):
#     # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29
#     bs = z.size(0)
#     z_dim = z.size(1)

#     # augment
#     z_permuted = z[torch.randperm(bs)]
#     alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
#     z_augmented = alpha * z + (1 - alpha) * z_permuted

#     # loss
#     v = torch.randn(bs, z_dim).to(z_augmented)
#     X, Jv = torch.autograd.functional.jvp(
#         decode, z_augmented, v=v, create_graph=True
#     )  # bs num_pts 3

#     Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
#     TrG = Jv_sq_norm.mean()

#     # vTG(z)v - vTv c
#     fm_loss = torch.mean(
#         (Jv_sq_norm - (torch.sum(v ** 2, dim=1)) * TrG / z_dim) ** 2
#     )
#     return fm_loss.sum()


def riemannian_latent_interpolation_continuous(
    model,
    Z0,
    Z1,
    z_template,
    n_geodesic_cps,
    snapshot_root,
    n_refinement_steps=1,
    n_random_point_samples=10,
    n_iters=1000,
):
    # model.pdm.set_sampler("ode", adjoint=True)
    Zs = continuous_geodesic(
        Z0,
        Z1,
        f_metric_tensor=lambda z: euclidean_metric_G(
            None,
            z,
            model,
            create_graph=True,
        ),
        # f_metric_tensor=lambda z: euclidean_metric_G(
        #     lambda Z: model.forward(
        #         Z,
        #         apply_pdm=True,
        #         apply_ltn=False,
        #         time_steps=5,
        #         n_samples=n_random_point_samples,
        #         adjoint_params=Z,
        #     )[1].view(Z.shape[0], -1, 3),
        #     z,
        #     model,
        #     create_graph=True,
        # ),
        # lambda Z: torch.randn(Z.shape[0], 100, 3).to(Z.device), z),
        verbose=True,
        n_iters=n_iters,
        n_cps=n_geodesic_cps,
    )

    # convert representations to shapes
    # Xs = torch.stack(
    #     [
    #         model.forward(Z, z=z_template, apply_ltn=False, apply_pdm=True)[1]
    #         for Z in Zs.unsqueeze(1)
    #     ]
    # )

    # plot the shapes
    # plot_shapes(Xs, snapshot_root)

    return Zs.cpu()


def interpolate(
    model,
    X0,
    X1,
    n_refinement_steps=0,
    n_random_point_samples=8192,
    device="cpu",
    snapshot_root=None,
    n_geodesic_cps=6,
    riemannian_kwargs={},
    Zs_train=None,
):
    # SETUP
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(X0, "face") else SubsetSample(8192)
    )

    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )

    f_sample_points_template = (
        tgt.SamplePoints(n_random_point_samples)
        if hasattr(X0, "face")
        else SubsetSample(n_random_point_samples)
    )
    T_template = tgt.Compose(
        [
            f_sample_points_template,
            tgt.NormalizeScale(),
        ]
    )

    # move data to device
    X0 = X0.to(device)
    X1 = X1.to(device)
    model = model.to(device)
    # embed shapes
    Z0, params0 = model.inverse(
        T(X0.clone()).pos, None, apply_stn=True, return_params=True, apply_ltn=False
    )
    Z1, params1 = model.inverse(
        T(X1.clone()).pos, None, apply_stn=True, return_params=True, apply_ltn=False
    )

    # setup template (choose one)
    X_template = T_template(X0.clone()).pos

    # template 1
    # z_template = model.pdm_inverse(X_template, condition=model.forward(Z0, apply_ltn=False), apply_stn=True)

    # template 2
    z_template = 0.7 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )

    # interpolate
    Zs_trajs = {}

    #linear_point_interpolation(model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps)

    Zs_trajs["Linear Interpolation"] = linear_latent_interpolation(
        model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps
    )

    Zs_trajs["Linear MS Interpolation"] = linear_metric_space_interpolation(
        model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps
    )

    # Zs_trajs[
    #     "Riemannian Interpolation Discrete"
    # ] = riemannian_latent_interpolation_discrete(
    #     model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps, **riemannian_kwargs
    # )
    # Zs_trajs[
    #     "Riemannian Interpolation Continuous"
    # ] = riemannian_latent_interpolation_continuous(
    #     model, Z0, Z1, z_template, n_geodesic_cps, snapshot_root, n_refinement_steps, **riemannian_kwargs
    # )

    # plot latent interpolation trajectories
    if Zs_train is not None:
        plot_interpolation_trajectory(Zs_trajs, Zs_train, snapshot_root)
