#!/usr/bin/env python3

import torch
import numpy as np

from math import cos, sin, pi
from torch_geometric.data import Data, InMemoryDataset
from pytorch_lightning.core.datamodule import LightningDataModule


class SyntheticGaussianDataset(InMemoryDataset, LightningDataModule):
    def generate_sample(self, t, n_point_samples):
        std = torch.Tensor([0.1, 0.1, 0.1])

        mean_g1 = torch.Tensor([cos(t * pi), sin(t * pi), 0])
        mean_g2 = torch.Tensor([cos(pi + t * pi), sin(pi + t * pi), 0])
        data = Data(
            pos=torch.cat(
                [
                    mean_g1 + std * torch.randn(int(n_point_samples / 2), 3),
                    mean_g2 + std * torch.randn(int(n_point_samples / 2), 3),
                ],
            ),
            id=[t],
        )

        return data

    def __init__(self, n_samples, n_point_samples, n_pc_resamples=10, **kwargs):
        super().__init__(**kwargs)

        np.random.seed(seed=24)

        if n_samples == 1:
            self.data, self.slices = self.collate(
                [self.generate_sample(0.5, n_point_samples) for s in range(10)]
            )

        else:
            self.data, self.slices = self.collate(
                [self.generate_sample(t / n_samples, n_point_samples) for s in range(n_pc_resamples) for t in range(n_samples)]
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

        np.random.seed(seed=24)

        if n_samples == 1:
            self.data, self.slices = self.collate(
                [self.generate_sample(0.5, n_point_samples) for s in range(10)]
            )
        else:
            self.data, self.slices = self.collate(
                [self.generate_sample(t / (n_samples - 1), n_point_samples) for s in range(10) for t in range(n_samples)]
            )


if __name__ == "__main__":
    import torch_geometric.transforms as tgt

    dataset = SyntheticGaussianDataset(
        n_samples=3,
        n_point_samples=8192,
        transform=tgt.Compose(
            [
                tgt.NormalizeScale(),
            ]
        ),
    )
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

    for data in dataset:
        plotter = Plotter()
        plotter.add_generic(data)
        plotter.show_bounds()
        plotter.camera_position = "xy"
        plotter.show()
