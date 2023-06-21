#!/usr/bin/env python3

import re
import os
import torch

from abc import ABC
from gembed import Configuration
from abc import ABC, abstractmethod
from pytorch_lightning.core.datamodule import LightningDataModule
from torch_geometric.data import InMemoryDataset

class AbstractDataset(ABC, InMemoryDataset, LightningDataModule):
    def __init__(
        self,
        root=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        root : str
            Root folder
        n_samples : int, optional
            Number of generative samples
        n_components : int, optional
            Number of components used to generate data
        """

        if root is None:
            path = Configuration()["Paths"]["DATA_DIR"]
            root = os.path.join(path, self.subdir)

        super().__init__(root=root, **kwargs)

        print(f"Loading dataset: {self}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def subdir(self) -> str:
        return re.sub('((?!^)(?<!_)[A-Z][a-z]+|(?<=[a-z0-9])[A-Z])', r'_\1', self.__class__.__name__).lower()


    @property
    @abstractmethod
    def _raw_data(self):
        ...

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # 1) get the in correspondence data
        data_list = self._raw_data

        # 2) preprocess the data
        data_list = [d for d in data_list if d is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 3) Store data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
