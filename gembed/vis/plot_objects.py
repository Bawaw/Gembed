#!/usr/bin/env python3

import pyvista as pv
from gembed.vis.plotter import Plotter


def plot_objects(
    *args,
    camera_position=[(-5, -5, 0), (0, 0, 0), (0, 0, 1)],
    snapshot_file_name=None,
    show_grid=False,
    theme="document",
    window_size=[1600, 2000],
    remove_scalar_bar=True,
    **kwargs
):
    """
    Plots 3D objects with optional scalar data.

    Parameters:
        *args: Variable length argument list containing tuples of shape and scalar data.
            - shape (object): The 3D object to be plotted.
            - scalars (array-like or None): Optional scalar data associated with the object.
        camera_position (list[tuple]): List of three tuples representing the camera's initial position in 3D space.
            Default: [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]
        snapshot_file_name (str): Optional filename to save the plot as a snapshot. If None, the plot is displayed interactively.
            Default: None
        show_grid (bool): Flag to indicate whether to display the grid in the plot.
            Default: False
        **kwargs: Additional keyword arguments to customize the plot appearance.

    Returns:
        None

    Notes:
        - The plot is rendered off-screen if a snapshot_file_name is provided.
        - The plot window size is set to [1600, 2000].
        - Each object in the args list is added to a separate subplot in the grid.
    """

    pv.set_plot_theme(theme)

    n_objects = len(args)
    plotter = Plotter(
        shape=(1, n_objects),
        off_screen=snapshot_file_name is not None,
    )
    plotter.window_size = window_size

    for i, (shape, scalars) in enumerate(args):
        plotter.subplot(0, i)
        plotter.add_generic(
            shape,
            scalars=scalars,
            render_points_as_spheres=True,
            point_size=5,
            **kwargs
        )
        if show_grid:
            plotter.show_grid()

    plotter.link_views()

    plotter.camera_position = camera_position

    if remove_scalar_bar and len(plotter._scalar_bars) > 0:
        plotter.remove_scalar_bar()

    if snapshot_file_name is not None:
        plotter.screenshot(snapshot_file_name, transparent_background=True)
    else:
        plotter.show()

    plotter.close()
