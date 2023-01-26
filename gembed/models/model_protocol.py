#!/usr/bin/env python3

from __future__ import annotations
from torch import Tensor
from torch.utils.data import Dataset
from typing import Protocol, Union


class ModelProtocol(Protocol):

    def fit(self, dataset: Union[Tensor, Dataset], **kwargs) -> ModelProtocol:
        ...

    @staticmethod
    def load(path: str, file_name: str, **kwargs) -> ModelProtocol:
        ...

    def save(self, path: str, file_name: str, **kwargs) -> None:
        ...
