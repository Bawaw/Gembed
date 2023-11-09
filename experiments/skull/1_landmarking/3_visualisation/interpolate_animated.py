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
import os


def linear_metric_space_interpolation(model, Z0, Z1, z_template, n_geodesic_cps):
    Z0_metric, Z1_metric = model.mtn.inverse(Z0), model.mtn.inverse(Z1)

    Zs_metric = torch.lerp(
        input=Z0_metric,
        end=Z1_metric,
        weight=torch.linspace(0, 1, n_geodesic_cps)[:, None].to(Z0.device),
    )

    Zts = model.mtn.forward(Zs_metric)

    # convert representations to shapes
    Xs = torch.stack(
        [
            model.forward(Zt, z=z_template, apply_ltn=False, apply_pdm=True)[1]
            for Zt in Zts.unsqueeze(1)
        ]
    )

    return Xs


def interpolate_animated(
    model,
    Zs,
    experiment_name,
    split,
    n_random_point_samples=8192,
    device="cpu",
    n_geodesic_cps=100,
    start_pause_frames=0,
    shape_pause_frames=0,
):
    # SETUP
    pl.seed_everything(42, workers=True)

    # move data to device
    z_template = 0.8 * model.pdm.base_distribution.sample(n_random_point_samples).to(
        device
    )
    z_scalars = z_template[:, 0].cpu()

    plotter = Plotter(off_screen=True)

    file_name = f"output/{experiment_name}/{split}/interpolation/animated/metric_space_interpolation.gif"

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    plotter.open_gif(file_name)

    plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]

    for i, Z0 in enumerate(Zs.unsqueeze(1)):
        Z1 = Zs[(i + 1) % len(Zs)]  # cycle back to start

        Xs = linear_metric_space_interpolation(
            model,
            Z0.to(device),
            Z1.to(device),
            z_template,
            n_geodesic_cps,
        )

        for t, X in enumerate(Xs):
            # if first frame
            if i == 0 and t == 0:
                pv_s1 = plotter.add_generic(
                    X.cpu(),
                    render_points_as_spheres=True,
                    scalars=z_scalars,
                    cmap="cool",
                )
                plotter.remove_scalar_bar()

                for _ in range(start_pause_frames + 1):
                    plotter.write_frame()

            else:
                plotter.update_coordinates(X.cpu())

                plotter.write_frame()

        for _ in range(shape_pause_frames):
            plotter.write_frame()

    plotter.close()
