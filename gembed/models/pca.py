#!/usr/bin/env python3

import os
from joblib import dump, load

import numpy as np
from sklearn.decomposition import PCA as PCASK

import torch
from gembed.core.module import InvertibleModule
from gembed.models import ModelProtocol
from torch import Tensor


class PCA(ModelProtocol, InvertibleModule):
    def __init__(self, n_components, seed=24, whiten=True):
        super().__init__()
        self._model = PCASK(n_components, random_state=seed, whiten=whiten)

    @property
    def n_components(self) -> int:
        return self._model.n_components_

    # AbstractInvertibleModule
    def forward(self, z: Tensor, **kwargs) -> Tensor:
        return torch.from_numpy(self.model.transform(z))

    def inverse(self, y: Tensor, **kwargs) -> Tensor:
        return torch.from_numpy(self.model.inverse_transform(y))

    # ModelProtocol
    def fit(self, dataset: Tensor, **kwargs) -> ModelProtocol:
        self.model = self._model.fit(dataset)
        print(
            f"PCA model fitted; explained_variance_ratio: {self.model.explained_variance_ratio_}, \n"
            f"resulting in a total sum of {self.model.explained_variance_ratio_.sum()}"
        )

        return self

    @staticmethod
    def load(path: str, file_name: str = "pca_model.joblib", **kwargs):
        load(os.path.join(path, file_name))

    def save(self, path: str, file_name="pca_model.joblib", **kwargs):
        dump(self, os.path.join(path, file_name))

    # DistributionProtocol
    def log_prob(self, data: Tensor) -> Tensor:
        return torch.from_numpy(self._model.score_samples(data))

    def sample(self, n_samples: int, seed=None):
        if seed is not None:
            np.random.seed(seed)

        z_samples = torch.randn(n_samples, self.n_components)
        return self.forward(z_samples)
