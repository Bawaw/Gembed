#!/usr/bin/env python3

import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects
from gembed.core.optim import gradient_ascent
from transform import SubsetSample
import pyvista as pv
from gembed.core.optim import gradient_ascent
from gembed.vis.plotter import Plotter


def reconstruct_animated(
    model,
    dataset,
    experiment_name,
    split,
    n_samples=10,
    n_point_samples=80000,
    device="cpu",
    sampled_vis_mesh=False,
    start_pause_frames=0,
    pre_refinement_pause_frames=0,
    end_pause_frames=0,
    n_refinement_steps=10,
):

    pl.seed_everything(42, workers=True)

    n_samples = min(n_samples, len(dataset))

    f_sample_points = lambda n_samples: (
        tgt.SamplePoints(n_samples)
        if hasattr(dataset[0], "face")
        else SubsetSample(n_samples)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points(8192),
            tgt.NormalizeScale(),
        ]
    )

    # point embedding for template mesh
    template_z = torch.randn(n_point_samples, 3)
    template_scalar = template_z[:, 1]

    model = model.to(device)
    template_z = template_z.to(device)

    for i, X_data in enumerate(dataset):
        X_data_transformed = T(X_data.clone()).to(device)

        print(f"Data id: {X_data.id}")

        # get the condition variable
        Z = model.inverse(X_data_transformed.pos, None, apply_stn=True)

        # reconstruct data sample in template format
        Xs_rec = model.forward(
            Z=Z, z=template_z, return_time_steps=True, apply_pdm=True
        )[1]

        Xs_rec = Xs_rec.cpu()

        # setup movie
        time_steps = len(Xs_rec)

        # initial state
        s1 = Xs_rec[0]

        plotter = Plotter(off_screen=True)
        pv_s1 = plotter.add_generic(
            s1, render_points_as_spheres=True, scalars=template_scalar, cmap="cool"
        )
        plotter.remove_scalar_bar()

        file_name = f"output/{experiment_name}/{split}/reconstruction/animated/{X_data.id[0]}.gif"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        plotter.open_gif(file_name)

        plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]

        # Run through each frame
        for _ in range(start_pause_frames + 1):
            plotter.write_frame()

        for t in range(1, time_steps):
            s1 = Xs_rec[t]
            plotter.update_coordinates(s1)

            plotter.write_frame()

        for _ in range(pre_refinement_pause_frames):
            plotter.write_frame()

        # final state
        s1 = Xs_rec[-1].to(device)

        for i in range(n_refinement_steps):
            s1 = gradient_ascent(
                init_x=s1,
                f_grad=lambda x, b, c: model.pdm.score(
                    x, torch.Tensor([0.0]).to(x.device), b, c
                ),
                condition=Z.clone(),
                batch_size=3000,
                n_steps=1,
            )

            plotter.update_coordinates(s1.detach().cpu())

            plotter.write_frame()

        for _ in range(end_pause_frames):
            plotter.write_frame()

        plotter.close()
