#!/usr/bin/env python3
import torch
import pytorch_lightning as pl
from torch_scatter import scatter_mean


class PointFlow(pl.LightningModule):
    def __init__(self, sdm, pdm):
        super().__init__()

        self.sdm = sdm
        self.pdm = pdm

    def pdm_inverse(self, **kwargs):
        return self.pdm.inverse(**kwargs)

    def pdm_forward(self, **kwargs):
        return self.pdm.forward(**kwargs)

    def inverse(self, x, batch):
        condition = self.sdm.inverse(x, batch)
        return condition

    # def forward(self, z, batch, z=None):
    #     condition = self.sdm.forward(z, batch)
    #     return condition

    def log_prob(self, x, batch):
        condition = self.sdm.inverse(x, batch)

        log_px = self.pdm.log_prob(x, condition, batch=batch)

        return log_px

    def training_step(self, train_batch, batch_idx):
        pos, batch = train_batch.pos, train_batch.batch

        log_px = self.log_prob(pos, batch)

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, train_batch.batch)
        nll = nll.mean()
        self.log("train_loss", nll, batch_size=train_batch.num_graphs)
        return nll

    def validation_step(self, valid_batch, batch_idx):
        pos, batch = valid_batch.pos, valid_batch.batch

        # log px_i
        log_px = self.log_prob(pos, batch)

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, valid_batch.batch)
        nll = nll.mean()
        self.log("valid_loss", nll, batch_size=valid_batch.num_graphs)
        return nll

    def configure_optimizers(self):
        # optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3)

        return optimiser


class AugmentedPointFlow(PointFlow):
    def __init__(
        self, sdm, pdm, train_sdm_augmentation=None, train_pdm_augmentation=None
    ):
        super().__init__(sdm, pdm)

        self.train_sdm_augmentation = train_sdm_augmentation
        self.train_pdm_augmentation = train_pdm_augmentation

    def training_step(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch.batch

        if self.train_sdm_augmentation is not None:
            x_augmented_sdm, batch_augmented_sdm = self.train_sdm_augmentation(
                x.clone(), batch.clone()
            )
        else:
            x_augmented_sdm, batch_augmented_sdm = x.clone(), batch.clone()

        if self.train_pdm_augmentation is not None:
            x_augmented_pdm, batch_augmented_pdm = self.train_pdm_augmentation(
                x.clone(), batch.clone()
            )
        else:
            x_augmented_pdm, batch_augmented_pdm = x.clone(), batch.clone()

        condition = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, condition, batch=batch_augmented_pdm
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log("train_loss", nll, batch_size=train_batch.num_graphs)
        return nll

    def validation_step(self, valid_batch, batch_idx):
        x, batch = valid_batch.pos, valid_batch.batch

        # log px_i
        if self.train_sdm_augmentation is not None:
            x_augmented_sdm, batch_augmented_sdm = self.train_sdm_augmentation(
                x.clone(), batch.clone()
            )
        else:
            x_augmented_sdm, batch_augmented_sdm = x.clone(), batch.clone()

        if self.train_pdm_augmentation is not None:
            x_augmented_pdm, batch_augmented_pdm = self.train_pdm_augmentation(
                x.clone(), batch.clone()
            )
        else:
            x_augmented_pdm, batch_augmented_pdm = x.clone(), batch.clone()

        condition = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, condition, batch=batch_augmented_pdm
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log("valid_loss", nll, batch_size=valid_batch.num_graphs)
        return nll

    def configure_optimizers(self):
        # optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3)

        return optimiser
