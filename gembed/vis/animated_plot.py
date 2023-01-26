#!/usr/bin/env python3

import torch
import pyvista as pv
from gembed.vis.plotter import Plotter
import pathlib

def animated_plot(time_sequence, file_name, camera_pos=None, reverse=False, scalars = None,
                  theme='document', show_grid = False, grid_args = {}, **kwargs):

    T = len(time_sequence)

    pv.set_plot_theme(theme)
    plotter = Plotter(off_screen=True)

    if camera_pos is not None:
        plotter.camera_position = camera_pos

    if show_grid:
        plotter.show_grid(**grid_args)

        # this is a workaround, because the bounds argument for
        # show_grid is not working. We add a transparant cube of size bounds
        # this line can be removed when the bug is fixed
        if "bounds" in grid_args:
            plotter.add_mesh(pv.Cube(bounds=grid_args['bounds']), opacity=0.)

    init_scalars = scalars[0] if scalars is not None else None
    main_shape = plotter.add_generic(
        time_sequence[0], scalars=init_scalars, **kwargs)

    plotter.show(auto_close=False)

    vid_format = pathlib.Path(file_name).suffix

    if vid_format == ".mp4":
        plotter.open_movie(file_name)
    if vid_format == ".gif":
        plotter.open_gif(file_name)

    for i in range(1, T):
        plotter.update_coordinates(time_sequence[i], mesh=main_shape)

        if scalars is not None and len(scalars) > 1:
            breakpoint()
            plotter.update_scalars(scalars[i])

        plotter.write_frame()

    if reverse:
        for i in range(T-1, 0, -1):

            plotter.update_coordinates(time_sequence[i], mesh=main_shape)

            if scalars is not None and len(scalars) > 1:
                plotter.update_scalars(scalars[i])

            plotter.write_frame()

    plotter.close()
