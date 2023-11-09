#!/usr/bin/env python3

import torch
from torch import nn

class ConcatFuse(nn.Module):
    def __init__(self, include_bias=False, *dims):
        super().__init__()

        dim_in = dims[:-1]
        dim_out = dims[-1]

        self.layers = nn.Linear(sum(dim_in), dim_out, bias=False)

    def forward(self, *args):
        return self.layers(torch.cat([*args], 1))
