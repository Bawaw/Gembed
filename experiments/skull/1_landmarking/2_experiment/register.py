#!/usr/bin/env python3

import os

import pytorch_lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects
from gembed.core.optim import gradient_ascent
from transform import SubsetSample


def register_template(
    model,
    dataset,
    template,
    device="cpu",
    plot_mesh_as_surface=False,
    snapshot_root=None,
    n_refinement_steps=0,
    n_template_samples=None,
):
    pl.seed_everything(42, workers=True)

    f_sample_points = lambda n_samples: (
        tgt.SamplePoints(n_samples)
        if hasattr(template, "face")
        else SubsetSample(n_samples)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points(8192),
            tgt.NormalizeScale(),
        ]
    )
    T_norm = tgt.Compose(
        [
            tgt.NormalizeScale(),
        ]
    )

    # transform template
    template = T_norm(template.clone())

    # resample the template
    if n_template_samples is not None:
        template_x = f_sample_points(n_template_samples)(
            template.clone()
        )  # draw new samples
    else:
        template_x = template.clone()

    template_scalar = template_x.pos[:, 0].cpu()

    # move data to gpu
    model = model.to(device)
    template = template.to(device)
    template_x = template_x.to(device)

    # point embedding for template mesh
    z_template = model.pdm_inverse(
        x=template_x.pos,
        # we sample the template for the condition
        condition=model.inverse(
            T(template.clone()).pos.to(device),
            batch=None,
            apply_stn=True,  # , z_id=template.id
        ),
        apply_stn=True,
    )

    ###### PLOT CLOUD EMBEDDING ######
    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.scatterplot(
    #     data={
    #         "x": z_template[:, 0].cpu(),
    #         "y": z_template[:, 1].cpu(),
    #     },
    #     x="x",
    #     y="y",
    # )
    # plt.show()
    # sns.scatterplot(
    #     data={
    #         "x": z_template[:, 0].cpu(),
    #         "z": z_template[:, 2].cpu(),
    #     },
    #     x="x",
    #     y="z",
    # )
    # plt.show()
    # z_template = torch.randn(int(1e4), 3).to(z_template.device)
    # scores = model.refinement_network(z_template)
    # plot_objects((z_template.cpu(), scores.cpu()), show_grid=True)
    # exit()
    ##################################

    for i, X_data in enumerate(dataset):
        X_data_transformed = T(X_data.clone()).to(device)

        print(f"Template id: {template.id}, Data id: {X_data.id}")

        # register template
        X_rec = template.clone()
        Z_rec, params = model.inverse(
            X_data_transformed.pos,
            batch=None,
            apply_stn=True,
            return_params=True,  # , z_id=X_data_transformed.id
        )
        X_rec.pos = model.pdm_forward(
            z=z_template,
            condition=Z_rec,
            return_time_steps=False,
        )

        # refine the registration
        X_rec_refined = X_rec.clone()
        if n_refinement_steps > 0:
            X_rec_refined.pos = gradient_ascent(
                init_x=X_rec_refined.pos.requires_grad_(True),
                f_grad=lambda x, b, c: model.pdm.score(
                    x, torch.Tensor([0.0]).to(x.device), b, c
                ),
                condition=Z_rec.clone(),
                batch_size=3000,
                n_steps=n_refinement_steps,
            ).detach()

        # # invert the spatial alignment
        if model.stn is not None:
            X_rec.pos = model.stn.inverse(X_rec.pos, None, params)
            X_rec_refined.pos = model.stn.inverse(X_rec_refined.pos, None, params)

        # move reconstruction to cpu
        X_rec = X_rec.cpu()
        X_data_transformed = X_data_transformed.cpu()
        X_rec_refined = X_rec_refined.cpu()
        template = template.cpu()

        # remove the surface information for plot
        if hasattr(X_data, "face") and not plot_mesh_as_surface:
            del X_data.face
            del X_rec.face
            del template_x.face
            del X_rec_refined.face

        if snapshot_root is not None:
            os.makedirs(snapshot_root, exist_ok=True)
            save_file_path = os.path.join(snapshot_root, f"random_shape_{i}.png")
        else:
            save_file_path = None

        # plot
        plot_objects(
            (template_x.cpu(), template_scalar),
            #(z_template.cpu(), template_scalar),
            (T_norm(X_data), None),
            (X_rec, template_scalar),
            (X_rec_refined, template_scalar),
            snapshot_file_name=save_file_path,
            window_size=[3 * 1600, 2000],
            cmap="cool",
            color="#cccccc",
        )
