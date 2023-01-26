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


class NormalisingFlow(
    pl.LightningModule, InvertibleModule, DistributionProtocol, ModelProtocol
):
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
        return_combined_dynamics: bool = False,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        # z=f(x), -log |det Jf|
        x, log_det_jac_f, *combined_dynamics = self.layers.forward(
            z=z, batch=batch, condition=condition, **kwargs
        )

        if not return_combined_dynamics:
            return x

        log_pz = self.base_distribution.log_prob(z)

        # (Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling and Inference." J. Mach. Learn. Res. 22.57 (2021): 1-64.)
        # log px = log pz - log |det Jf|
        #        => log pz + (-log |det Jf|)
        log_px = log_pz + log_det_jac_f

        # concat output
        return [x, log_px] + combined_dynamics

    def inverse(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        return_combined_dynamics: bool = False,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        # z=f_inv(x), -log |det Jf_inv|
        z, log_det_jac_f_inv, *combined_dynamics = self.layers.inverse(
            x=x, batch=batch, condition=condition, **kwargs
        )

        if not return_combined_dynamics:
            return z

        log_pz = self.base_distribution.log_prob(z)

        # (Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling and Inference." J. Mach. Learn. Res. 22.57 (2021): 1-64.)
        # log px = log pz + log |det Jf_inv|
        #        => log pz - (-log |det Jf_inv|)
        #        => log pz + div(Jf_inv)
        log_px = log_pz - log_det_jac_f_inv

        # concat output
        return [z, log_px] + combined_dynamics

    # DISTRIBUTION_PROTOCOL
    def sample(
        self,
        n_samples: int,
        condition: Union[Tensor, None] = None,
        seed: Union[int, None] = None,
        return_combined_dynamics: bool = True,
        **kwargs
    ) -> Union[List[Tensor], Tensor]:

        z, *_ = self.base_distribution.sample(n_samples, seed)

        return self.forward(
            z, condition, return_combined_dynamics=return_combined_dynamics, **kwargs
        )

    def log_prob(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
    ):

        _, log_px, *_ = self.inverse(
            x, batch=batch, condition=condition, return_combined_dynamics=True
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
