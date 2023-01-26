#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
import torch.distributions as tdist
from gembed.core.distribution import DistributionProtocol

class MultivariateNormal(nn.Module, DistributionProtocol):
    def __init__(self, loc, scale):
        super().__init__()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def dist(self):
        return tdist.MultivariateNormal(self.loc, self.scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self,
               n_samples: int,
               seed: Union[int, None] = None) -> Tensor:

        sample_shape = torch.Size([n_samples])
        return self.dist.sample(sample_shape)
