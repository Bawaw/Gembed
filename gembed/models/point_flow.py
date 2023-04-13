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

        log_px = self.pdm.log_prob(x, batch=batch, condition=condition)

        return log_px

    def log_prob_grad(self, x, batch):
        log_px = self.log_prob(x, batch)

        # TODO: check if this is correct
        return torch.autograd.grad(log_px.sum(), x, create_graph=True)[0]

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
    """TODO: Augmented Point Flow with decoupled data lanes."""

    def __init__(
        self,
        sdm,
        pdm,
        stn=None,
    ):
        super().__init__(sdm, pdm)

        self.stn = stn

    def pdm_inverse(self, x, batch=None, **kwargs):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().pdm_inverse(x=x, batch=batch, **kwargs)

    def inverse(self, x, batch=None):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().inverse(x=x, batch=batch)

    # def training_step(self, train_batch, batch_idx):
    def training_step(self, data, batch_idx):
        (x_augmented_sdm, batch_augmented_sdm) = data[0].pos, data[0].batch
        (x_augmented_pdm, batch_augmented_pdm) = data[1].pos, data[1].batch

        if self.stn is not None:
            x_augmented_sdm = self.stn(x_augmented_sdm, batch_augmented_sdm)
            x_augmented_pdm = self.stn(x_augmented_pdm, batch_augmented_pdm)

        condition = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, batch=batch_augmented_pdm, condition=condition
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log(
            "train_loss",
            nll,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        return nll

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, data, batch_idx):
        (x_augmented_sdm, batch_augmented_sdm) = data[0].pos, data[0].batch
        (x_augmented_pdm, batch_augmented_pdm) = data[1].pos, data[1].batch

        # apply spatial transformer to the data
        if self.stn is not None:
            x_augmented_sdm = self.stn(x_augmented_sdm, batch_augmented_sdm)
            x_augmented_pdm = self.stn(x_augmented_pdm, batch_augmented_pdm)

        # log px_i
        condition = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, batch=batch_augmented_pdm, condition=condition
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log(
            "valid_loss",
            nll,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        return nll

    def configure_optimizers(self):
        # optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

        optimiser = torch.optim.AdamW(
            [
                {"params": self.sdm.parameters(), "weight_decay": 1e-2},
                {"params": self.pdm.parameters(), "weight_decay": 1e-2},
                {"params": self.stn.parameters()},
            ],
            weight_decay=0,
            lr=1e-3,
        )

        return optimiser


class SingleLaneAugmentedPointFlow(PointFlow):
    """TODO: Augmented Point Flow"""

    def __init__(
        self,
        sdm,
        pdm,
        stn=None,
    ):
        super().__init__(sdm, pdm)

        self.stn = stn

    def pdm_inverse(self, x, batch=None, **kwargs):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().pdm_inverse(x=x, batch=batch, **kwargs)

    def inverse(self, x, batch=None):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().inverse(x=x, batch=batch)

    # def training_step(self, train_batch, batch_idx):
    def training_step(self, data, batch_idx):
        (x_augmented, batch_augmented) = data.pos, data.batch

        if self.stn is not None:
            x_augmented = self.stn(x_augmented, batch_augmented)

        condition = self.sdm.inverse(x_augmented, batch_augmented)
        log_px = self.pdm.log_prob(
            x_augmented, batch=batch_augmented, condition=condition
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented)
        nll = nll.mean()
        self.log(
            "train_loss",
            nll,
            batch_size=data.num_graphs,
            sync_dist=True,
        )
        return nll

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, data, batch_idx):
        (x_augmented, batch_augmented) = data.pos, data.batch

        if self.stn is not None:
            x_augmented = self.stn(x_augmented, batch_augmented)

        condition = self.sdm.inverse(x_augmented, batch_augmented)
        log_px = self.pdm.log_prob(
            x_augmented, batch=batch_augmented, condition=condition
        )

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented)
        nll = nll.mean()
        self.log(
            "valid_loss",
            nll,
            batch_size=data.num_graphs,
            sync_dist=True,
        )
        return nll

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # optimiser = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        # optimiser = torch.optim.AdamW(
        #     self.parameters(),
        #     weight_decay=1e-4,
        #     lr=1e-3,
        # )
        # optimiser = torch.optim.AdamW(
        #     [
        #         {"params": self.pdm.parameters(), "weight_decay": 1e-2},
        #         {"params": self.sdm.parameters()},
        #     ],
        #     weight_decay=0,
        #     lr=1e-3,
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=100, factor=0.9, verbose=True, min_lr=1e-5
        )

        # optimiser = torch.optim.AdamW(
        #     [
        #         {"params": self.sdm.parameters(), "weight_decay": 1e-2},
        #         {"params": self.pdm.parameters(), "weight_decay": 1e-2},
        #         {"params": self.stn.parameters()},
        #     ],
        #     weight_decay=0,
        #     lr=1e-3,
        # )

        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }


class AugmentedPointFlow_3(PointFlow):
    """TODO: Augmented Point Flow with decoupled data lanes."""

    def __init__(
        self,
        sdm,
        pdm,
        stn=None,
    ):
        super().__init__(sdm, pdm)

        self.stn = stn

    def pdm_inverse(self, x, batch=None, **kwargs):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().pdm_inverse(x=x, batch=batch, **kwargs)

    def inverse(self, x, batch=None):
        if self.stn is not None:
            x = self.stn(x, batch)

        return super().inverse(x=x, batch=batch)

    # def training_step(self, train_batch, batch_idx):
    def training_step(self, data, batch_idx):
        (x_augmented_sdm, batch_augmented_sdm) = data[0].pos, data[0].batch
        (x_augmented_pdm, batch_augmented_pdm) = data[1].pos, data[1].batch

        if self.stn is not None:
            x_augmented_sdm = self.stn(x_augmented_sdm, batch_augmented_sdm)
            x_augmented_pdm = self.stn(x_augmented_pdm, batch_augmented_pdm)

        condition_sdm = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        condition_pdm = self.sdm.inverse(x_augmented_pdm, batch_augmented_pdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, batch=batch_augmented_pdm, condition=condition_sdm
        )

        mse = (condition_pdm - condition_sdm).pow(2).sum(-1)
        mse = mse.mean()

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log(
            "train_loss",
            nll,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        self.log(
            "train_mse",
            mse,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )

        loss = nll + mse
        self.log(
            "new_train_loss",
            loss,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        return loss

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, data, batch_idx):
        (x_augmented_sdm, batch_augmented_sdm) = data[0].pos, data[0].batch
        (x_augmented_pdm, batch_augmented_pdm) = data[1].pos, data[1].batch

        if self.stn is not None:
            x_augmented_sdm = self.stn(x_augmented_sdm, batch_augmented_sdm)
            x_augmented_pdm = self.stn(x_augmented_pdm, batch_augmented_pdm)

        condition_sdm = self.sdm.inverse(x_augmented_sdm, batch_augmented_sdm)
        condition_pdm = self.sdm.inverse(x_augmented_pdm, batch_augmented_pdm)
        log_px = self.pdm.log_prob(
            x_augmented_pdm, batch=batch_augmented_pdm, condition=condition_sdm
        )

        mse = (condition_pdm - condition_sdm).pow(2).sum(-1)
        mse = mse.mean()

        # MLE
        # -1/N sum_i log px_i
        nll = -scatter_mean(log_px, batch_augmented_pdm)
        nll = nll.mean()
        self.log(
            "valid_loss",
            nll,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        self.log(
            "valid_mse",
            mse,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )

        loss = nll + mse
        self.log(
            "new_valid_loss",
            loss,
            batch_size=data[0].num_graphs,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        # optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

        optimiser = torch.optim.AdamW(
            [
                {"params": self.sdm.parameters(), "weight_decay": 1e-2},
                {"params": self.pdm.parameters(), "weight_decay": 1e-2},
                {"params": self.stn.parameters()},
            ],
            weight_decay=0,
            lr=1e-3,
        )

        return optimiser
