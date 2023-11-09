#!/usr/bin/env python3

import os
from typing import List, Union

import pytorch_lightning as pl
import torch
from gembed.core.distribution import DistributionProtocol
from gembed.core.module import InvertibleModule
from gembed.models.model_protocol import ModelProtocol
from torch import Tensor
from torch.utils.data import Dataset
from torch_scatter import scatter_mean


class NormalisingFlow(pl.LightningModule, InvertibleModule, DistributionProtocol):
    def __init__(
        self, base_distribution: DistributionProtocol, layers: InvertibleModule
    ):

        super().__init__()
        self.base_distribution = base_distribution
        self.layers = layers

    # ABSTRACT_INVERTIBLE_MODULE
    def forward(
        self,
        z: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        include_combined_dynamics: bool = False,
        include_log_density: bool = False,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        x, d_log_p, *combined_dynamics = self.layers.forward(
            z=z, batch=batch, condition=condition,
            include_combined_dynamics=include_combined_dynamics,
            include_change_in_log_density=include_log_density,
            **kwargs
        )

        output = (x,)

        if include_log_density:
            log_pz = self.base_distribution.log_prob(z)
            log_px = log_pz + d_log_p
            output += (log_px,)

        if include_combined_dynamics:
            output += tuple(combined_dynamics)

        # concat output
        return output[0] if len(output) == 1 else output

    def inverse(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        include_combined_dynamics: bool = False,
        include_log_density: bool = False,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        # z=f_inv(x), -log |det Jf_inv|
        z, d_log_p, *combined_dynamics = self.layers.inverse(
            x=x, batch=batch, condition=condition,
            include_combined_dynamics=include_combined_dynamics,
            include_change_in_log_density=include_log_density,
            **kwargs
        )

        output = (z,)

        if include_log_density:
            log_pz = self.base_distribution.log_prob(z)
            log_px = log_pz + d_log_p
            output += (log_px,)

        if include_combined_dynamics:
            output += tuple(combined_dynamics)

        # concat output
        return output[0] if len(output) == 1 else output

    # DISTRIBUTION_PROTOCOL
    def sample(
        self,
        n_samples: int,
        condition: Union[Tensor, None] = None,
        seed: Union[int, None] = None,
        include_combined_dynamics: bool = True,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        z, *_ = self.base_distribution.sample(n_samples, seed)

        return self.forward(
            z, condition, include_combined_dynamics=include_combined_dynamics, **kwargs
        )

    def log_prob(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
    ):

        _, log_px, *_ = self.inverse(
            x, batch=batch, condition=condition, include_combined_dynamics=True, include_log_density=True
        )

        return log_px

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """Returns the Negative log likelihood of the batch which is used for training.

        L(\theta) = -\frac{1}{N} \sum_i=0^N log p_\theta (x_i)

        """

        # log px_i
        log_px = self.log_prob(train_batch.pos)

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, train_batch.batch)
        self.log("train_loss", nll)
        return nll

    def fit(
        self, dataset: Dataset, gpus: List = [1], max_epochs: int = 100, **kwargs
    ) -> ModelProtocol:

        trainer = pl.Trainer(gpus=gpus, limit_train_batches=100, max_epochs=max_epochs)
        trainer.fit(model=self, train_dataloaders=dataset.train_dataloader)
        return self

    def load(self, path: str, file_name: str, **kwargs) -> ModelProtocol:
        return self.load_from_checkpoint(os.path.join(path, file_name), **kwargs)

    def save(self, path: str, file_name: str, **kwargs) -> None:
        raise NotImplementedError()
