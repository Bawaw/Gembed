#!/usr/bin/env python3

import torch
import torch.nn as nn


class ConcatSquashLinear(nn.Module):
    """Source: https://github.com/stevenygd/PointFlow/blob/master/models/diffeq_layers.py"""

    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
