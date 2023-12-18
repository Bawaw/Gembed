#!/usr/bin/env python3
import sys

sys.path.insert(0, "../")

import pyvista as pv
from transform import ExcludeIDs, ExtractSurfacePCByThreshold, SubsetSample, SwapAxes
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from torch_geometric.transforms import Center, Compose
from gembed.utils.dataset import train_valid_test_split


def delaunay_triangulation(template, pv_template):
    return pv_template.delaunay_2d()


def surface_reconstruction(template, pv_template):
    return pv_template.reconstruct_surface()


def openai_sdf_model(template, pv_template):
    import torch
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.util.pc_to_mesh import marching_cubes_mesh
    from point_e.util.plotting import plot_point_cloud
    from point_e.util.point_cloud import PointCloud

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("creating SDF model...")
    name = "sdf"
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print("loading SDF model...")
    model.load_state_dict(load_checkpoint(name, device))

    coords = SubsetSample(15000)(template.clone()).pos.numpy()
    channel = torch.zeros(coords.shape[0]).numpy()
    pc = PointCloud(coords=coords, channels={"R": channel, "G": channel, "B": channel})

    mesh = marching_cubes_mesh(
        pc=pc, model=model, batch_size=128, grid_size=32, progress=True
    )

    with open("mesh.ply", "wb") as f:
        mesh.write_ply(f)
    mesh = pv.get_reader("mesh.ply").read()
    return mesh


dataset = ParisVolumetricSkulls(
    pre_filter=ExcludeIDs([15]),
    pre_transform=Compose(
        [ExtractSurfacePCByThreshold(), SwapAxes([2, 1, 0]), Center()]
    ),
)


train, valid, test = train_valid_test_split(dataset)

template = train[0]
pv_template = pv.PolyData(template.pos.numpy())

# pv_template_reconstruction = delaunay_triangulation(template, pv_template)
# pv_template_reconstruction = surface_reconstruction(template, pv_template)
pv_template_reconstruction = openai_sdf_model(template, pv_template)

# plot
pv.set_plot_theme("dark")
plotter = pv.Plotter(shape=[1, 2])

plotter.subplot(0, 0)
plotter.add_mesh(pv_template, render_points_as_spheres=True, point_size=5, cmap="cool")

plotter.subplot(0, 1)
plotter.add_mesh(pv_template_reconstruction, cmap="cool")

plotter.link_views()
plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]
plotter.show()
