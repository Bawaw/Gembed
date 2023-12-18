#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Union
from torch import Tensor

from torch.nn import Module


class InvertibleModule(Module):
    def forward(self, *input, **kwargs) -> Union[List[Tensor], Tensor]:
        raise NotImplementedError()

    def inverse(self, *input, **kwargs) -> Union[List[Tensor], Tensor]:
        raise NotImplementedError()
