#!/usr/bin/env python3

from typing import Union

import torch
import torch.distributions as tdist
import torch.nn as nn
from torch import Tensor

from gembed.core.distribution import DistributionProtocol


class MultivariateNormal(nn.Module, DistributionProtocol):
    """The `MultivariateNormal` class is a PyTorch module that represents a multivariate normal
    distribution and provides methods for calculating log density and sampling from the
    distribution.
    """

    def __init__(self, loc, scale):
        """
        The function initializes the loc and scale attributes of an object.

        :param loc: The `loc` parameter represents the mean of the distribution. It is a scalar value or a
        tensor of the same shape as the input data. It determines the center of the distribution
        :param scale: The "scale" parameter represents the scaling factor for the distribution. It is used
        to control the spread or dispersion of the distribution. A larger scale value will result in a wider
        distribution, while a smaller scale value will result in a narrower distribution
        """

        super().__init__()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def dist(self):
        """
        The function `dist` returns a multivariate normal distribution of the object.
        :return: The `dist` method is returning a `torch.distribution.MultivariateNormal` object.
        """
        return tdist.MultivariateNormal(self.loc, self.scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, n_samples: int, seed: Union[int, None] = None) -> Tensor:
        sample_shape = torch.Size([n_samples])
        return self.dist.sample(sample_shape)
