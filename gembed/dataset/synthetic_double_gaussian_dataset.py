#!/usr/bin/env python3

import torch
import numpy as np

from math import cos, sin, pi, sqrt
from torch_geometric.data import Data, InMemoryDataset
from lightning import LightningDataModule
import lightning as pl


# class SyntheticGaussianDataset(InMemoryDataset, LightningDataModule):
#     def generate_sample(self, z, n_point_samples):
#         z1, z2 = z
#         # std = 0.01 + 0.05 * z2
#         std = 0.01 + 0.5 * z2

#         mean_g1 = torch.Tensor([cos(z1 * (pi / 2)), sin(z1 * (pi / 2)), 0])
#         mean_g2 = torch.Tensor([cos(pi + z1 * (pi / 2)), sin(pi + z1 * (pi / 2)), 0])

#         data = Data(
#             pos=torch.cat(
#                 [
#                     mean_g1 + std * torch.randn(int(n_point_samples / 2), 3),
#                     mean_g2 + std * torch.randn(int(n_point_samples / 2), 3),
#                 ],
#             ),
#             id=torch.stack([z1, z2]),
#         )
#         return data

#     def __init__(self, n_samples, n_point_samples, n_pc_resamples=10, **kwargs):
#         super().__init__(**kwargs)

#         pl.seed_everything(42, workers=True)

#         if n_samples == 1:
#             self.data, self.slices = self.collate(
#                 [
#                     self.generate_sample(
#                         torch.Tensor([0.5]), torch.Tensor([0.5]), n_point_samples
#                     )
#                     for s in range(10)
#                 ]
#             )

#         else:
#             zs = (0.5 + 0.2 * torch.randn(n_samples, 2)).clamp(0, 1)
#             self.data, self.slices = self.collate(
#                 [
#                     self.generate_sample(z, n_point_samples)
#                     for s in range(1 + n_pc_resamples)
#                     for z in zs
#                 ]
#             )


class SyntheticGaussianDataset(InMemoryDataset, LightningDataModule):
    def generate_sample(self, z, n_point_samples):
        z1, z2 = z

        # std = 0.01 + 0.05 * z2
        std = 0.1 * z2

        r = 0.5
        mean_g1 = torch.Tensor([r * cos(z1 * (pi / 2)), r * sin(z1 * (pi / 2)), 0])
        mean_g2 = torch.Tensor(
            [r * cos(pi + z1 * (pi / 2)), r * sin(pi + z1 * (pi / 2)), 0]
        )

        # subtract corner samples
        n_point_samples -= 8 * 10

        data = Data(
            pos=torch.cat(
                [
                    mean_g1 + std * torch.randn(int(n_point_samples / 2), 3),
                    mean_g2 + std * torch.randn(int(n_point_samples / 2), 3),
                    # torch.Tensor([1, 0, 0]) + 0.01 * torch.randn(10, 3),
                    # torch.Tensor([-1, 0, 0]) + 0.01 * torch.randn(10, 3),
                    # torch.Tensor([0, 1, 0]) + 0.01 * torch.randn(10, 3),
                    # torch.Tensor([0, -1, 0]) + 0.01 * torch.randn(10, 3),
                    # torch.Tensor([0, 0, 1]) + 0.01 * torch.randn(10, 3),
                    # torch.Tensor([0, 0, -1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([1, 1, 1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([1, 1, -1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([1, -1, 1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([-1, 1, 1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([-1, -1, 1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([-1, 1, -1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([1, -1, -1]) + 0.01 * torch.randn(10, 3),
                    torch.Tensor([-1, -1, -1]) + 0.01 * torch.randn(10, 3),
                ],
            ),
            id=torch.stack([z1, z2])[None],
        )
        return data

    def __init__(self, n_samples, n_point_samples, n_pc_resamples=10, **kwargs):
        super().__init__(**kwargs)

        pl.seed_everything(42, workers=True)

        # zs ∈ [0, 1]
        zs = (0.5 + 0.2 * torch.randn(n_samples, 2)).clamp(0, 1)
        self.data, self.slices = self.collate(
            [
                self.generate_sample(z, n_point_samples)
                for s in range(1 + n_pc_resamples)
                for z in zs
            ]
        )


class SyntheticGaussianDataset2(InMemoryDataset, LightningDataModule):
    def generate_sample(self, t, n_point_samples):
        std = torch.Tensor([0.1, 0.1, 0.1])

        r = t
        n_gaussians = 6
        mean_gs = [
            torch.Tensor(
                [
                    r * cos((i / n_gaussians) * 2 * pi),
                    r * sin((i / n_gaussians) * 2 * pi),
                    0,
                ]
            )
            for i in range(n_gaussians)
        ]

        data = Data(
            pos=torch.cat(
                [
                    mean_g + std * torch.randn(int(n_point_samples / n_gaussians), 3)
                    for mean_g in mean_gs
                ]
            ),
            id=[t],
        )

        return data

    def __init__(self, n_samples, n_point_samples, n_pc_resamples=10, **kwargs):
        super().__init__(**kwargs)

        pl.seed_everything(42, workers=True)

        if n_samples == 1:
            self.data, self.slices = self.collate(
                [self.generate_sample(0.5, n_point_samples) for s in range(10)]
            )
        else:
            self.data, self.slices = self.collate(
                [
                    self.generate_sample(t / (n_samples - 1), n_point_samples)
                    for s in range(10)
                    for t in range(n_samples)
                ]
            )


if __name__ == "__main__":
    import torch_geometric.transforms as tgt

    dataset = SyntheticGaussianDataset(
        n_samples=100,
        n_point_samples=8192,
        n_pc_resamples=0,
        transform=tgt.Compose(
            [
                tgt.NormalizeScale(),
            ]
        ),
    )
    # dataset = SyntheticGaussianDataset(
    #     n_samples=1,
    #     n_point_samples=8192,
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #         ]
    #     ),
    # )
    # dataset = SyntheticGaussianDataset2(
    #     n_samples=3,
    #     n_point_samples=8192,
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #         ]
    #     ),
    # )
    from gembed.vis.plotter import Plotter

    # for data in dataset:
    #     plotter = Plotter()
    #     plotter.add_generic(data)
    #     plotter.show_bounds()
    #     plotter.camera_position = "xy"
    #     plotter.show()

    # from gembed.vis.plotter import Plotter

    plotter = Plotter()
    # for i, data in enumerate(dataset[5:10]):  # manually picked 9 nice examples
    for i, data in enumerate(dataset):  # manually picked 9 nice examples
        plotter.add_generic(
            data,
            scalars=i * torch.ones(data.pos.shape[0]),
            render_points_as_spheres=True,
            cmap="Set1",
        )
    plotter.show_bounds()
    plotter.camera_position = "xy"
    plotter.show()
