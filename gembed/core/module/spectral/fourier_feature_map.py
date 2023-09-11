#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn as nn


class FourierFeatureMap(nn.Module):
    r""" Models the dynamics."""

    def __init__(
        self, in_channels, out_channels, fourier_feature_scale=None
    ):
        super().__init__()

        if fourier_feature_scale is None:
            fourier_feature_scale = 2 ** -2

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        self.scale_w = fourier_feature_scale
        self.Wx = nn.Parameter(
            self.scale_w * torch.randn(out_channels // 2, in_channels),
            requires_grad=False,
        )

    def gaussian_encoding(self, v, b):
        r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
            b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`
        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`
        See :class:`~rff.layers.GaussianEncoding` for more details.

        Source: https://github.com/jmclong/random-fourier-features-pytorch/tree/main/rff
        """
        vp = 2 * np.pi * v @ b.T
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

    def forward(self, x):
        x = self.gaussian_encoding(x, self.Wx)
        return x


# if __name__ == "__main__":
#     import pyvista as pv

#     for i in range(-4, 4):
#         ffs = 2 ** i
#         f = FourierFeatureMap(2, 2, ffs)
#         fc = FourierFeatureMap(2, 2, ffs, center=True)

#         vtk = lambda x: pv.PolyData(
#             torch.concat([x, torch.zeros(x.shape[0], 1)], 1).numpy()
#         )

#         identity_grid = torch.stack(
#             torch.meshgrid(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20)), -1
#         ).view(-1, 2)

#         scalars = identity_grid[:, 0]
#         fourier_grid = f(identity_grid.clone())
#         fourier_grid_2 = fc(identity_grid.clone())

#         plotter = pv.Plotter(shape=(1, 3))
#         plotter.subplot(0, 0)
#         plotter.add_mesh(
#             vtk(identity_grid),
#             scalars=scalars,
#             point_size=15,
#             render_points_as_spheres=True,
#         )
#         plotter.show_grid()

#         plotter.subplot(0, 1)
#         plotter.add_mesh(
#             vtk(fourier_grid),
#             scalars=scalars,
#             point_size=15,
#             render_points_as_spheres=True,
#         )
#         plotter.show_grid()

#         plotter.subplot(0, 2)
#         plotter.add_mesh(
#             vtk(fourier_grid_2),
#             scalars=scalars,
#             point_size=15,
#             render_points_as_spheres=True,
#         )
#         plotter.show_grid()

#         plotter.link_views()

#         plotter.show()
