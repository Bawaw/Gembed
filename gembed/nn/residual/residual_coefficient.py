#!/usr/bin/env python3

import torch
import torch.nn as nn

class ResidualCoefficient(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = torch.nn.parameter.Parameter(
            data=torch.zeros(1), requires_grad=True
        )

    def forward(self, x):
        return self.alpha * x
