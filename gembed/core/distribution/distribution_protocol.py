#!/usr/bin/env python3

from torch import Tensor
from typing import Protocol, Union, List


class DistributionProtocol(Protocol):
    def log_prob(self, x: Tensor, **kwargs)-> Tensor:
        ...

    def sample(self, n_samples: int, seed: Union[int, None] = None, **kwargs) -> Union[List[Tensor], Tensor]:
        ...
