#!/usr/bin/env python3

import torch
import numpy as np
import pyvista as pv

from torch_geometric.data import InMemoryDataset
from gembed.utils.adapter import vtk_to_torch_geometric_data
from pytorch_lightning.core.datamodule import LightningDataModule


class SyntheticPyramidDataset(InMemoryDataset, LightningDataModule):
    def generate_pyramid(self, index):
        eps1 = np.random.uniform(-0.5, +0.5)
        eps2 = np.random.uniform(-0.5, +0.5)
        eps3 = np.random.uniform(-0.2, +0.2)
        pyramid = (
            pv.Pyramid(
                [
                    [1 + eps1, 1, 0],
                    [-1 + eps1, 1, 0],
                    [-1 - eps1, -1, 0],
                    [1 - eps1, -1, 0],
                    [0, 0, 0.5 + eps3],
                ]
            )
            .triangulate()
            .extract_surface()
        )

        # pyramid.plot(show_edges=True)

        data = vtk_to_torch_geometric_data(pyramid)
        data.pos = data.pos.float()
        data.id = [index]

        return data

    def __init__(self, n_samples, **kwargs):
        super().__init__(**kwargs)

        np.random.seed(seed=24)
        self.data, self.slices = self.collate(
            [self.generate_pyramid(i) for i in range(n_samples)]
        )


# if __name__ == "__main__":
#     import torch_geometric.transforms as tgt

#     dataset = SyntheticPyramidDataset(
#         n_samples=500,
#         transform=tgt.Compose(
#             [
#                 tgt.NormalizeScale(),
#                 tgt.SamplePoints(512),
#             ]
#         ),
#     )


#     from gembed.vis.plotter import Plotter
#     for data in dataset:
#         plotter = Plotter()
#         plotter.add_generic(data)
#         plotter.show_bounds()
#         plotter.camera_position = "xy"
#         plotter.show()
