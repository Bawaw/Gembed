#!/usr/bin/env python3

import torch
from torch_geometric.transforms import BaseTransform


class Clip(BaseTransform):
    def __init__(self, threshold=0.5):
        assert (
            threshold < 1 and threshold > 0
        ), f"Clip expected threshold value to be within bound [0, 1], but was {threshold}"

        self.threshold = threshold

    def __call__(self, data):
        rest = 1 - self.threshold

        data.pos = data.pos[data.pos[:, 2] > -(self.threshold + rest * torch.rand(1))]
        data.pos = data.pos[data.pos[:, 2] < self.threshold + rest * torch.rand(1)]
        return data
