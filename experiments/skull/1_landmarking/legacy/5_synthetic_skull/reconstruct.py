#!/usr/bin/env python3

import torch
import lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects

from transform import SubsetSample


def sampled_reconstruction(
    model,
    dataset,
    n_samples=10,
    n_point_samples=80000,
    device="cpu",
    sampled_vis_mesh=False,
):
    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(4096),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(4096),
            ]
        )

    # point embedding for template mesh
    template_z = torch.randn(n_point_samples, 3)
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

        # evaluate the log probability for each point
        # log_px = model.pdm.log_prob(
        #     X_rec.clone(), batch=None, condition=Z
        # )

        # X_rec = X_rec[log_px > -1]
        #
        X_rec = X_rec.cpu()
        X_data = X_data.cpu()

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(n_point_samples)(X_data)

        # plot
        plot_objects((X_data, X_data.pos[:, 1]), (X_rec, X_rec[:, 1]), cmap="cool")


def transferred_reconstruction(model, dataset, template, n_samples=10, device="cpu"):
    pl.seed_everything(42, workers=True)

    template = tgt.Center()(template.clone())

    n_samples = min(n_samples, len(dataset))

    # T_template = SubsetSample(4096)
    T_template = tgt.Compose([])

    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(4096),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(4096),
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

        # if model.stn is not None:
        #     params = model.stn.get_transform_params(X_data_transformed.pos, None)
        #     X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

        X_rec = X_rec.cpu()
        X_data = X_data.cpu()

        # plot
        plot_objects((X_data, X_data.pos[:, 1]), (X_rec, template_scalar), cmap="cool")


def template_reconstruction(
    model,
    dataset,
    template,
    vis_template,
    n_samples=10,
    device="cpu",
    sampled_vis_mesh=False,
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
                tgt.SamplePoints(4096),
            ]
        )
    else:
        T_template = tgt.Compose(
            [
                SubsetSample(4096),
            ]
        )
    if hasattr(dataset[0], "face"):
        T = tgt.Compose(
            [
                tgt.SamplePoints(4096),
            ]
        )
    else:
        T = tgt.Compose(
            [
                SubsetSample(4096),
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

        X_data = X_data.cpu()
        X_rec = X_rec.cpu()

        if hasattr(X_data, "face") and sampled_vis_mesh:
            X_data = tgt.SamplePoints(X_rec.pos.shape[0])(X_data)

        # plot
        plot_objects((X_data, X_data.pos[:, 0]), (X_rec, template_scalar), cmap="cool")


# def reconstruct_shape(model, dataset, template, n_samples=10):
#     print("WARNING: not using STN")

#     n_samples = min(n_samples, len(dataset))

#     # point embedding for template mesh
#     template_z = model.pdm_inverse(
#         x=template.pos, condition=model.inverse(template.pos, None)
#     )

#     for i in range(n_samples):
#         # grab data sample
#         X_data = dataset[i].clone()


#         # reconstruct data sample in template format
#         X_rec = template.clone()
#         X_rec.pos = model.pdm_forward(
#             z=template_z,
#             condition=model.inverse(X_data.pos, None),
#             return_time_steps=False,
#         )

#         # MSE
#         rec_error = (X_data.pos - X_rec.pos).pow(2).sum(-1)

#         # plot
#         plot_objects((X_data, torch.zeros_like(rec_error)), (X_rec, rec_error), cmap="cool")

# def reconstruct_shape_from_sampled_pc(model, dataset, template, n_samples=10):
#     print("WARNING: not using STN")
#     n_samples = max(n_samples, len(dataset))
#     dataset = [dataset[i].clone() for i in range(n_samples)]

#     # point sampler for encoder
#     T = SamplePoints(2048)

#     # (shape, shape embeddings)
#     shape_and_embedding = [(d, sp := T(d.clone()).pos, model.inverse(sp, None)) for d in dataset]

#     # point embedding for template mesh
#     template_z = model.pdm_inverse(
#         x=template.pos, condition=model.inverse(T(template.clone()).pos, None)
#     )

#     for X_data, X_data_sampled, Z in shape_and_embedding:
#         # reconstruct shape from template point embedding and shape condition
#         X_rec = template.clone()
#         X_rec.pos = model.pdm_forward(
#             z=template_z, condition=Z, return_time_steps=False
#         )

#         # MSE
#         rec_error = (X_data.pos - X_rec.pos).pow(2).sum(-1)

#         # plot
#         plot_objects((X_data, torch.zeros_like(rec_error)), (X_rec, rec_error), cmap="cool")

# def reconstruct_shape(model, dataset, template, n_samples=10):
#     n_samples = min(n_samples, len(dataset))

#     # point embedding for template mesh
#     template_z = model.pdm_inverse(
#         x=template.pos, condition=model.inverse(template.pos, None)
#     )

#     for i in range(n_samples):
#         # grab data sample
#         X_data = dataset[i].clone()

#         # template_z = model.pdm_inverse(
#         #     x=X_data.pos, condition=model.inverse(X_data.pos, None)
#         # )

#         # reconstruct data sample in template format
#         X_rec = template.clone()
#         X_rec.pos = model.pdm_forward(
#             z=template_z,
#             condition=model.inverse(X_data.pos, None),
#             return_time_steps=False,
#         )

#         # invert the spatial transformation so that result matches input coordinates
#         params = model.stn.get_transform_params(X_data.pos, None)
#         X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

#         # MSE
#         rec_error = (X_data.pos - X_rec.pos).pow(2).sum(-1)

#         # plot
#         plot_objects((X_data, None), (X_rec, rec_error))


# def reconstruct_shape_from_sampled_pc(model, dataset, template, n_samples=10):
#     n_samples = max(n_samples, len(dataset))
#     dataset = [dataset[i].clone() for i in range(n_samples)]

#     # point sampler for encoder
#     T = SamplePoints(2048)

#     # (shape, shape embeddings)
#     shape_and_embedding = [(d, sp := T(d.clone()).pos, model.inverse(sp, None)) for d in dataset]

#     # point embedding for template mesh
#     template_z = model.pdm_inverse(
#         x=template.pos, condition=model.inverse(T(template.clone()).pos, None)
#     )

#     for X_data, X_data_sampled, Z in shape_and_embedding:
#         # reconstruct shape from template point embedding and shape condition
#         X_rec = template.clone()
#         X_rec.pos = model.pdm_forward(
#             z=template_z, condition=Z, return_time_steps=False
#         )

#         # invert the spatial transformation so that result matches input coordinates
#         params = model.stn.get_transform_params(X_data_sampled, None)
#         X_rec.pos = model.stn.inverse(X_rec.pos, None, params)

#         # MSE
#         rec_error = (X_data.pos - X_rec.pos).pow(2).sum(-1)

#         # plot
#         plot_objects((X_data, None), (X_rec, rec_error))
