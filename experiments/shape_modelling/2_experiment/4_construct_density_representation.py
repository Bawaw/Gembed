#!/usr/bin/env python3
import os
import torch
import seaborn as sns
import pyvista as pv
import lightning as pl
import matplotlib.pyplot as plt
import torch_geometric.transforms as tgt

from math import sqrt
from gembed.vis import Plotter
from helper import load_experiment, pathcat
from matplotlib.collections import LineCollection
from gembed.utils.adapter import torch_geomtric_data_to_vtk

def plot_result_2D(log_px_grid, data_transformed, intersection_plane, depth, file_path):
    ax = sns.heatmap(log_px_grid, xticklabels=False, yticklabels=False)
    ax.invert_yaxis()

    if hasattr(data_transformed, "face"):

        def compute_intersection(data_transformed, intersection_plane, return_dict):
            mesh = torch_geomtric_data_to_vtk(
                data_transformed.pos, data_transformed.face
            )
            
            intersection, _, _ = mesh.intersection(intersection_plane)
            lines = torch.from_numpy(intersection.lines).view(-1, 3)
            points = log_px_grid.shape[0] * ((intersection.points[:, :2] + 1) / 2)

            intersection_curve = LineCollection(
                [[points[i], points[j]] for (_, i, j) in lines], colors="cornflowerblue", label="GT"
            )

            return_dict["intersection_curve"] = intersection_curve

        # compute intersection can get stuck so launched in a different process
        import multiprocessing
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=compute_intersection, name="Intersection", args=(data_transformed, intersection_plane, return_dict))
        p.start()
        p.join(240)

        # kill the process
        p.terminate() 
        if p.is_alive():
            p.kill() # if process is still alive force kill it 
        p.join()

        if "intersection_curve" in return_dict:
            ax.add_collection(return_dict["intersection_curve"])
            ax.legend()

        
    else:
        # select thin (determined by atol) slice of point cloud at depth
        mask = torch.isclose(
            data_transformed.pos[:, -1], torch.tensor(depth), atol=0.01
        )
        pc_slice = data_transformed.pos[mask, :-1]

        # plot volume grid intersection
        pc_slice = log_px_grid.shape[0] * ((pc_slice + 1) / 2)  # rescale slice to fit heatmap

        sns.scatterplot(
            x=pc_slice[:, 0],
            y=pc_slice[:, 1],
            s=1,
            marker="s",
            linewidth=0,
        )


    if file_path is None:
        plt.show()
    else:
        os.makedirs(file_path, exist_ok=True)
        ax.get_figure().savefig(pathcat(file_path, "density.png"), bbox_inches="tight", dpi=300)

    plt.close()


# def plot_result_2D(log_px_grid, data_transformed, intersection_plane, depth, file_path):
#     ax = sns.heatmap(log_px_grid)
#     ax.invert_yaxis()

#     if hasattr(data_transformed, "face"):
#         mesh = torch_geomtric_data_to_vtk(
#             data_transformed.pos, data_transformed.face
#         )
        
#         intersection, _, _ = mesh.intersection(intersection_plane)
#         lines = torch.from_numpy(intersection.lines).view(-1, 3)
#         points = log_px_grid.shape[0] * ((intersection.points[:, :2] + 1) / 2)

#         intersection_curve = LineCollection(
#             [[points[i], points[j]] for (_, i, j) in lines], colors="cornflowerblue", label="GT"
#         )

#         # plot mesh grid intersection
#         ax.add_collection(intersection_curve)
#         ax.legend()
#     else:
#         # select thin (determined by atol) slice of point cloud at depth
#         mask = torch.isclose(
#             data_transformed.pos[:, -1], torch.tensor(depth), atol=0.01
#         )
#         pc_slice = data_transformed.pos[mask, :-1]

#         # plot volume grid intersection
#         pc_slice = log_px_grid.shape[0] * ((pc_slice + 1) / 2)  # rescale slice to fit heatmap

#         sns.scatterplot(
#             x=pc_slice[:, 0],
#             y=pc_slice[:, 1],
#             s=1,
#             marker="s",
#             linewidth=0,
#         )


#     if file_path is None:
#         plt.show()
#     else:
#         os.makedirs(file_path, exist_ok=True)
#         ax.get_figure().savefig(pathcat(file_path, "density.png"), bbox_inches="tight", dpi=300)

#     plt.close()

def plot_result_3D(x_grid, log_px, data_transformed, intersection_plane, depth):
    if hasattr(data_transformed, "face"):
        mesh = torch_geomtric_data_to_vtk(
            data_transformed.pos, data_transformed.face
        )
        intersection, _, _ = mesh.intersection(intersection_plane)
        lines = torch.from_numpy(intersection.lines).view(-1, 3)
        points = int(sqrt(log_px.shape[0])) * ((intersection.points[:, :2] + 1) / 2)

        intersection_curve = LineCollection(
            [[points[i], points[j]] for (_, i, j) in lines]
        )

        # plot results 3D
        plotter = Plotter()
        plotter.add_generic(mesh, opacity=0.5)
        plotter.add_generic(intersection_plane, color="black")
        plotter.add_generic(x_grid.cpu(), scalars=log_px.cpu(), cmap="plasma")
        plotter.add_generic(intersection, color="blue")
        plotter.show_grid()
        plotter.view_xy()
        plotter.show()

    else:
        # select thin (determined by atol) slice of point cloud at depth
        mask = torch.isclose(
            data_transformed.pos[:, -1], torch.tensor(depth), atol=0.01
        )
        pc_slice = data_transformed.pos[mask, :-1]

        plotter = Plotter()
        plotter.add_generic(data_transformed.pos, opacity=0.01)
        plotter.add_generic(x_grid.cpu(), scalars=log_px.cpu(), cmap="plasma")
        plotter.add_generic(pc_slice, color="blue")
        plotter.show_grid()
        plotter.view_xy()
        plotter.show()

def run(
    model,
    T_sample,
    dataset,
    depth=0.0,
    grid_size=128,
    plane="xy",
    plot_log_px=True,
    device="cpu",
    file_path=None
):
    pl.seed_everything(42, workers=True)

    # setup transforms
    T_norm = tgt.NormalizeScale()

    # create pixel density grid
    grid = torch.stack(
        torch.meshgrid(
            [torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)],
            indexing="xy",
        ),
        -1,
    )
    (x1, x2), x_3 = grid.view(-1, 2).T, depth * torch.ones(grid_size ** 2)

    if plane == "xy":
        # convert pixel grid to point cloud
        x_grid = torch.stack([x1, x2, x_3], -1).to(device)

        # construct plane [-1, 1]^2
        intersection_plane = pv.Plane(
            center=(0, 0, depth), direction=(0, 0, 1), i_size=2, j_size=2
        ).triangulate()

    else:
        raise NotImplementedError()

    for i, data in enumerate(dataset):
        # get data representation
        data_transformed = T_norm(data.clone())

        Z, stn_params = model.inverse(
            T_sample(data_transformed.clone()).pos.to(device),
            batch=None,
            apply_stn=True,
            return_params=True,
        )

        # get density per point in grid
        log_px = model.log_prob(
            x_grid.clone(), batch=None, condition=Z, apply_stn=True, stn_params=stn_params
        )

        # convert density in 3D-coords back to grid
        if not plot_log_px:
            log_px = torch.exp(log_px)
        log_px_grid = log_px.view(grid_size, grid_size).cpu()

        # plot
        plot_result_2D(log_px_grid, data_transformed, intersection_plane, depth, file_path=pathcat(file_path, str(data.id)))
        #plot_result_3D(x_grid, log_px, data_transformed, intersection_plane, depth)

def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])


    with torch.no_grad():
        run(model, T_sample, train[:5], device=device, file_path=pathcat(file_path, "train"))
        run(model, T_sample, valid[:5], device=device, file_path=pathcat(file_path, "valid"))
        run(model, T_sample, test[:5], device=device, file_path=pathcat(file_path, "test"))

if __name__ == "__main__":
    main()