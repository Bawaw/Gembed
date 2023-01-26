#!/usr/bin/env python3

import pyvista as pv
from gembed.vis.plotter import Plotter

def plot_objects(*args, **kwargs):
    """Expected format (data, scalars) or (data, None)"""

    pv.set_plot_theme("dark")

    n_objects = len(args)
    plotter = Plotter(shape=(1, n_objects))
    plotter.window_size = [1600, 2000]

    for i, (shape, scalars) in enumerate(args):
        plotter.subplot(0, i)
        plotter.add_generic(
            shape,
            scalars=scalars,
            render_points_as_spheres=True,
            point_size=5,
            **kwargs
        )
        plotter.show_grid()

    plotter.link_views()

    plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]
    plotter.show()
