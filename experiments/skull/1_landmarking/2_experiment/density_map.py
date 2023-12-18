#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

import lightning as pl
import pyvista as pv
import torch
import torch_geometric.transforms as tgt
from gembed.utils.adapter import torch_geomtric_data_to_vtk
from gembed.vis import Plotter
from transform import SubsetSample


def density_map(
    model,
    dataset,
    depth=0.0,
    grid_size=128,  # 512,  # 500,
    plane="xy",
    device="cpu",
    snapshot_root=None,
    plot_log_px=False
):
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(dataset[0], "face") else SubsetSample(8192)
    )

    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )

    # create pixel density grid
    grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)],
            indexing="xy",
        ),
        -1,
    )
    (x1, x2), x_0 = grid.view(-1, 2).T, depth * torch.ones(grid_size ** 2)

    if plane == "xy":
        # convert pixel grid to point cloud
        x_grid = torch.stack([x1, x2, x_0], -1)

        intersection_plane = pv.Plane(
            center=(0, 0, depth), direction=(0, 0, 1), i_size=2, j_size=2
        ).triangulate()

    else:
        raise NotImplementedError()

    # transfer objects to GPU
    model = model.to(device)
    x_grid = x_grid.to(device)

    for i, data in enumerate(dataset):
        print(f"Data id: {data.id}")

        # compute intersection curve
        data_transformed = tgt.NormalizeScale()(data.clone())

        # get density per point
        Z, params = model.inverse(
            T(data.clone()).pos.to(device),
            batch=None,
            apply_stn=True,
            return_params=True,
        )
        log_px = model.log_prob(
            x_grid.clone(), batch=None, condition=Z, apply_stn=True, stn_params=params
        )

        # convert density from coords to grid
        # log_px = torch.sigmoid(log_px)
        # log_px = torch.clamp(log_px, 0, 100)
        if not plot_log_px:
            log_px = torch.exp(log_px)
        log_px_grid = log_px.view(grid_size, grid_size).cpu()

        # plot the results 2D
        ax = sns.heatmap(log_px_grid)
        ax.invert_yaxis()

        # if hasattr(data_transformed, "face"):
        #     mesh = torch_geomtric_data_to_vtk(
        #         data_transformed.pos, data_transformed.face
        #     )
        #     intersection, _, _ = mesh.intersection(intersection_plane)
        #     lines = torch.from_numpy(intersection.lines).view(-1, 3)
        #     points = grid_size * ((intersection.points[:, :2] + 1) / 2)

        #     intersection_curve = LineCollection(
        #         [[points[i], points[j]] for (_, i, j) in lines]
        #     )

        #     # plot results 3D
        #     # pv.set_plot_theme("dark")

        #     # plotter = Plotter()
        #     # plotter.add_generic(mesh, opacity=1.0)
        #     # plotter.add_generic(intersection_plane, color="black")
        #     # plotter.add_generic(x_grid.cpu(), scalars=log_px.cpu(), cmap="plasma")
        #     # plotter.add_generic(intersection, color="blue")
        #     # plotter.show_grid()
        #     # plotter.view_xy()
        #     # plotter.show()

        #     # plot mesh grid intersection
        #     ax.add_collection(intersection_curve)
        # else:
        #     # select thin (determined by atol) slice of point cloud at depth
        #     mask = torch.isclose(
        #         data_transformed.pos[:, -1], torch.tensor(depth), atol=0.01
        #     )
        #     pc_slice = data_transformed.pos[mask, :-1]

        #     # plot results 3D
        #     # pv.set_plot_theme("dark")

        #     # plotter = Plotter()
        #     # plotter.add_generic(data_transformed.pos, opacity=0.01)
        #     # plotter.add_generic(x_grid.cpu(), scalars=log_px.cpu(), cmap="plasma")
        #     # plotter.add_generic(pc_slice, color="blue")
        #     # plotter.show_grid()
        #     # plotter.view_xy()
        #     # plotter.show()

        #     # plot volume grid intersection
        #     pc_slice = grid_size * ((pc_slice + 1) / 2)  # rescale slice to fit heatmap
        #     sns.scatterplot(
        #         x=pc_slice[:, 0],
        #         y=pc_slice[:, 1],
        #         s=1,
        #         marker="s",
        #         linewidth=0,
        #     )

        plt.show()
