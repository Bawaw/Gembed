#!/usr/bin/env python3

from torch import Tensor
from gembed.core.module import InvertibleModule


class LinearTransformation(InvertibleModule):
    def __init__(self, transformation_matrix: Tensor):
        super().__init__()
        self.M = transformation_matrix

    def forward(self, pos: Tensor, **kwargs) -> Tensor:
        return pos @ self.M

    def inverse(self, pos: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError()
