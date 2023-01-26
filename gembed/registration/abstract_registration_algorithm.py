#!/usr/bin/env python3

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional, Tuple
from abc import ABC, abstractmethod

class AbstractRegistrationAlgorithm(ABC):

    @abstractmethod
    def _estimate_transform(self, fixed : Data, moving : Data) -> Tensor:
        pass

    @abstractmethod
    def _transform(self, moving: Data, param: Tensor) -> Data:
        pass

    def __call__(self, fixed: Data, moving: Data, return_parametrisation=False) -> Tuple[
            Data, Optional[Tensor]]:

        _moving, _fixed = moving.clone(), fixed.clone()

        param = self._estimate_transform(_fixed, _moving)
        result = self._transform(moving, param)

        if return_parametrisation:
            return result, param

        return result
