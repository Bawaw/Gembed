#!/usr/bin/env python3

import torch
from torch import nn


class LinearCombination(nn.Module):
    def __init__(self, *dims, include_bias=False):
        super().__init__()

        dims_in = dims[:-1]
        dim_out = dims[-1]

        self.layers = nn.ModuleList()
        for dim_in in dims_in:
            self.layers.append(nn.Linear(dim_in, dim_out, bias=False))

            # orthogonal initialisation
            # _layer = nn.Linear(dim_in, dim_out, bias=False)
            # torch.nn.init.orthogonal_(_layer.weight)
            # self.layers.append(_layer)

        self.include_bias = include_bias

        if include_bias:
            self.bias = nn.parameter.Parameter(
                data=torch.empty(dim_out), requires_grad=True
            )
            self.bias.data.zero_()

    def forward(self, *inputs):
        # remove last input (batch)
        inputs = inputs[:-1]

        res = self.layers[0](inputs[0])

        # W1x1 + W2x2 + ... + Wnxn
        for i in range(1, len(inputs)):
            res += self.layers[i](inputs[i])

        if self.include_bias:
            res += self.bias

        return res
