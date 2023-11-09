#!/usr/bin/env python3

import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects
from gembed.core.optim import gradient_ascent
from transform import SubsetSample
import pyvista as pv
from gembed.core.optim import gradient_ascent
from gembed.vis.plotter import Plotter


def sampled_reconstruction(
    model,
    dataset,
    n_point_samples=80000,
    sampled_vis_mesh=False,
    n_refinement_steps=0,
    device="cpu",
):
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(dataset[0], "face") else SubsetSample(8192)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )
    T_norm = tgt.Compose(
        [
            tgt.NormalizeScale(),
        ]
    )

    # point embedding for template mesh
    template_z = model.pdm.base_distribution.sample(n_point_samples)
    #    template_z = 0.7 * model.pdm.base_distribution.sample(n_point_samples)
    # mask = model.refinement_network(template_z).squeeze(1)
    # template_z = template_z[mask]
    # template_z = model.pdm.base_distribution.sample(n_point_samples)
    template_scalar = template_z[:, 1]

    model = model.to(device)
    template_z = template_z.to(device)

    for i, X_data in enumerate(dataset):
        # grab data sample
        X_data_transformed = T(X_data.clone()).to(device)
        X_data = T_norm(X_data)

        print(f"Index: {i}, Data id: {X_data.id}")

        # get the condition variable
        Z, params = model.inverse(
            X_data_transformed.pos, None, apply_stn=True, return_params=True
        )

        # reconstruct data sample in template format
        X_rec = model.pdm_forward(
            z=template_z,
            condition=Z,
            return_time_steps=False,
        )

        # refine the point cloud
        if n_refinement_steps > 0:
            X_rec.pos = gradient_ascent(
                init_x=X_rec.requires_grad_(True),
                f_grad=model.log_prob_grad,
                condition=Z.clone(),
                batch_size=3000,  # 7000,
                n_steps=n_refinement_steps,
            ).detach()

        # invert the spatial alignment
        if model.stn is not None:
            X_rec = model.stn.inverse(X_rec, None, params)

        # evaluate the log probability for each point
        # log_px = model.pdm.log_prob(
        #     X_rec.clone(), batch=None, condition=Z
        # )

        # X_rec = X_rec[log_px > -1]

        X_rec = X_rec.cpu()
        X_data = X_data.cpu()

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(n_point_samples)(X_data)

        # plot
        import pyvista as pv
        from gembed.utils import adapter

        X_mesh_rec = adapter.vtk_to_torch_geometric_data(
            pv.PolyData(X_rec.numpy()).delaunay_2d(tol=1e-2)
        )
        plot_objects(
            (X_data, X_data.pos[:, 1]),
            (X_rec, X_rec[:, 1]),
            (X_mesh_rec, X_mesh_rec.pos[:, 1]),
            cmap="cool",
        )


def sampled_reconstruction_with_correction(
    model,
    dataset,
    n_samples=10,
    # n_point_samples=80000,
    n_point_samples=20000,
    device="cpu",
    sampled_vis_mesh=False,
    n_input_samples=4096,
    refinement_sampler=None,
):
    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )

    # point embedding for template mesh
    template_z = torch.randn(n_point_samples, 3)
    # template_z = 0.7 * torch.randn(n_point_samples, 3)
    # template_scalar = template_z[:, 0]

    model = model.to(device)
    template_z = template_z.to(device)

    for i in range(n_samples):
        # grab data sample
        X_data = tgt.Center()(dataset[i].clone())
        X_data_transformed = T(X_data.clone()).to(device)
        X_data = X_data.to(device)

        print(f"Data id: {X_data.id}")

        # get the condition variable
        Z = model.inverse(X_data_transformed.pos, None)

        # reconstruct data sample in template format
        X_rec = model.pdm_forward(
            z=template_z,
            condition=Z,
            return_time_steps=False,
        )
        if model.stn is not None:
            params = model.stn.get_transform_params(X_data_transformed.pos, None)
            X_rec = model.stn.inverse(X_rec, None, params)

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(n_point_samples)(X_data)

        plots = [
            (X_data.cpu(), X_data.pos.cpu()[:, 1]),
            (X_rec.cpu(), X_rec.cpu()[:, 1]),
        ]

        if refinement_sampler is not None:
            X_rec_cor_1 = refinement_sampler(
                init_x=X_rec.clone().requires_grad_(True),
                f_grad=model.log_prob_grad,
                condition=Z.clone(),
                batch_size=2096,  # 7000,
                n_steps=5,
            ).detach()

            # if model.stn is not None:
            #     params = model.stn.get_transform_params(X_data_transformed.pos, None)
            #     X_rec_cor_1 = model.stn.inverse(X_rec_cor_1, None, params)

            X_rec_cor_1 = X_rec_cor_1.cpu()

            plots.append((X_rec_cor_1, X_rec_cor_1[:, 1]))

            # X_rec_cor_2 = refinement_sampler(
            #     X_rec.clone().requires_grad_(True),
            #     model.log_prob_grad,
            #     Z.clone(),
            #     model,
            #     n_steps=100,
            # ).detach().cpu()
            # plots.append((X_rec_cor_1, X_rec_cor_1[:, 1]))

            # X_rec_cor_3 = refinement_sampler(
            #     X_rec.clone().requires_grad_(True),
            #     model.log_prob_grad,
            #     Z.clone(),
            #     model,
            #     n_steps=1000,
            # ).detach().cpu()
            # plots.append((X_rec_cor_1, X_rec_cor_1[:, 1]))

        # plot
        plot_objects(*plots, cmap="cool")


def transferred_reconstruction(
    model,
    dataset,
    template,
    n_samples=10,
    device="cpu",
    n_input_samples=4096,
):
    pl.seed_everything(42, workers=True)

    template = tgt.Center()(template.clone())

    n_samples = min(n_samples, len(dataset))

    # T_template = SubsetSample(n_input_samples)
    T_template = tgt.Compose([])

    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )

    model = model.to(device)
    transformed_template = T_template(template.clone()).to(device)
    template_scalar = template.pos[:, 1]
    template = template.to(device)

    # point embedding for template mesh
    template_z = model.pdm_inverse(
        x=template.pos, condition=model.inverse(transformed_template.pos, None)
    )

    for i in range(n_samples):
        # grab data sample
        X_data = tgt.Center()(dataset[i].clone())
        X_data_transformed = T(X_data.clone()).to(device)
        X_data = X_data.to(device)

        print(f"Template id: {template.id}, Data id: {X_data.id}")

        # reconstruct data sample in template format
        X_rec = template.clone()
        X_rec.pos = model.pdm_forward(
            z=template_z,
            condition=model.inverse(X_data_transformed.pos, None),
            return_time_steps=False,
        )

        if model.stn is not None:
            params = model.stn.get_transform_params(X_data_transformed.pos, None)
            X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

        X_rec = X_rec.cpu()
        X_data = X_data.cpu()

        # plot
        plot_objects((X_data, X_data.pos[:, 1]), (X_rec, template_scalar), cmap="cool")


def template_reconstruction_with_correction(
    model,
    dataset,
    template,
    vis_template,
    n_samples=10,
    device="cpu",
    sampled_vis_mesh=False,
    n_input_samples=4096,
    refinement_sampler=None,
):
    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    # center templates
    mean = template.pos.mean(dim=-2, keepdim=True)
    template.pos = template.pos - mean
    vis_template.pos = vis_template.pos - mean

    if hasattr(template, "face"):
        T_template = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T_template = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )
    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )

    model = model.to(device)
    transformed_template = T_template(template.clone()).to(device)
    template_scalar = vis_template.pos[:, 0]
    template = template.to(device)
    vis_template = vis_template.to(device)

    # point embedding for template mesh
    template_z = model.pdm_inverse(
        x=vis_template.pos, condition=model.inverse(transformed_template.pos, None)
    )

    for i in range(n_samples):
        # grab data sample
        X_data = tgt.Center()(dataset[i].clone())
        X_data_transformed = T(X_data.clone()).to(device)
        X_data = X_data.to(device)

        print(
            f"Template id: {template.id}, Mesh Template id: {vis_template.id}, Data id: {X_data.id}"
        )

        # reconstruct data sample in template format
        X_rec = vis_template.clone()
        Z = model.inverse(X_data_transformed.pos, None)
        X_rec.pos = model.pdm_forward(
            z=template_z,
            condition=Z,
            return_time_steps=False,
        )

        if model.stn is not None:
            params = model.stn.get_transform_params(X_data_transformed.pos, None)
            X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(X_rec.pos.shape[0])(X_data)

        plots = [
            (X_data.clone().cpu(), X_data.pos.cpu()[:, 0]),
            (X_rec.clone().cpu(), template_scalar),
        ]

        if refinement_sampler is not None:
            X_rec_cor_1 = X_rec.clone()

            X_rec_cor_1.pos = refinement_sampler(
                init_x=X_rec.pos.clone().requires_grad_(True),
                f_grad=model.log_prob_grad,
                condition=Z.clone(),
                n_steps=5,
            ).detach()

            if model.stn is not None:
                params = model.stn.get_transform_params(X_data_transformed.pos, None)
                X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

            X_rec_cor_1 = X_rec_cor_1.cpu()
            plots.append((X_rec_cor_1, template_scalar))

        # from gembed.utils.adapter import torch_geomtric_data_to_vtk
        # X_rec_cor_4 = torch_geomtric_data_to_vtk(
        #     X_rec_cor_1.pos, X_rec_cor_1.face
        # ).smooth_taubin(n_iter=50, pass_band=0.4)

        # plot
        plot_objects(*plots, cmap="cool")


def template_reconstruction(
    model,
    dataset,
    template,
    vis_template,
    n_samples=10,
    device="cpu",
    sampled_vis_mesh=False,
    n_input_samples=4096,
):
    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    # center templates
    mean = template.pos.mean(dim=-2, keepdim=True)
    template.pos = template.pos - mean
    vis_template.pos = vis_template.pos - mean

    if hasattr(template, "face"):
        T_template = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T_template = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )
    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(n_input_samples),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(n_input_samples),
            ]
        )

    model = model.to(device)
    transformed_template = T_template(template.clone()).to(device)
    template_scalar = vis_template.pos[:, 0]
    template = template.to(device)
    vis_template = vis_template.to(device)

    # point embedding for template mesh
    template_z = model.pdm_inverse(
        x=vis_template.pos, condition=model.inverse(transformed_template.pos, None)
    )

    for i in range(n_samples):

        # grab data sample
        X_data = tgt.Center()(dataset[i].clone())
        X_data_transformed = T(X_data.clone()).to(device)
        X_data = X_data.to(device)

        print(
            f"Template id: {template.id}, Mesh Template id: {vis_template.id}, Data id: {X_data.id}"
        )

        # reconstruct data sample in template format
        X_rec = vis_template.clone()
        X_rec.pos = model.pdm_forward(
            z=template_z,
            condition=model.inverse(X_data_transformed.pos, None),
            return_time_steps=False,
        )

        if model.stn is not None:
            params = model.stn.get_transform_params(X_data_transformed.pos, None)
            X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

        X_data = X_data.cpu()
        X_rec = X_rec.cpu()

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(X_rec.pos.shape[0])(X_data)

        # plot
        plot_objects((X_data, X_data.pos[:, 0]), (X_rec, template_scalar), cmap="cool")
