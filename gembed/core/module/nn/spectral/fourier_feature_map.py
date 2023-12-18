#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn


class FourierFeatureMap(nn.Module):
    """The `FourierFeatureMap` class models the dynamics by applying a Gaussian encoding to the input
    tensor using a randomly initialized projection matrix."""

    def __init__(self, in_channels, out_channels, fourier_feature_scale):
        super().__init__()

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
