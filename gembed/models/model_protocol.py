#!/usr/bin/env python3

from __future__ import annotations
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from typing import Protocol, Union


class ModelProtocol(Protocol):

    def fit(self, model : ModelProtocol, train_loader : DataLoader, valid_loader: DataLoader, experiment_name : str, **kwargs) -> ModelProtocol:
        ...

    @staticmethod
    def load(model_name: str, version : int, **kwargs) -> ModelProtocol:
        ...

    def save(self, path: str, file_name: str, **kwargs) -> None:
        ...
