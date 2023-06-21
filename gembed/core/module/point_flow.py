#!/usr/bin/env python3
import torch
import pytorch_lightning as pl
from torch_scatter import scatter_mean, scatter_sum


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

    def log_prob(self, x, batch, condition=None):
        if condition is None:
            condition = self.sdm.inverse(x, batch)

        log_px = self.pdm.log_prob(x, batch=batch, condition=condition)

        return log_px

    def log_prob_grad(self, x, batch, condition=None):
        """
        Compute the gradient of the log probability with respect to the input `x`.

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            condition (Optional[torch.Tensor]): The condition tensor. Default is None.

        Returns:
            torch.Tensor: The gradient of the log probability with respect to `x`.

        """

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            log_px = self.log_prob(x, batch, condition)

            # TODO: check that log_px is indeed a scalar
            nabla_log_px = torch.autograd.grad(log_px.sum(), x, create_graph=True)[0]

        return nabla_log_px

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
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

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
        if self.stn is None:
            optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            # optimiser = torch.optim.Adam(
            #     [
            #         {"params": self.sdm.parameters()},
            #         {"params": self.pdm.parameters(), "weight_decay": 1e-4},
            #         # encourage simple transformations and simple function for stn
            #         {"params": self.stn.parameters(), "weight_decay": 1e-4},
            #     ],
            #     weight_decay=0,
            #     lr=1e-3,
            #     eps=1e-4,
            # )
            optimiser = torch.optim.AdamW(
                self.parameters(), lr=5e-4, eps=1e-3, weight_decay=1e-4
            )

        optim_dict = {"optimizer": optimiser}

        # plateau scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimiser, patience=100, factor=0.9, verbose=True, min_lr=1e-5
        # )
        # optim_dict["lr_scheduler"] = {
        #     "scheduler": scheduler,
        #     "monitor": "regularised_train_loss",
        # }

        # 10k big steps and then smaller to prevent diverging
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimiser,
        #     lr_lambda=lambda step: 0.9 if step >= 1e4 else 1.0,
        #     verbose=True,
        # )

        # optim_dict["lr_scheduler"] = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        # }

        return optim_dict


class RegularisedSingleLaneAugmentedPointFlow(SingleLaneAugmentedPointFlow):
    """TODO: Augmented Point Flow"""

    def __init__(self, lambda_e=0.01, lambda_n=0.01, **kwargs):
        super(RegularisedSingleLaneAugmentedPointFlow, self).__init__(**kwargs)

        self.lambda_e = lambda_e
        self.lambda_n = lambda_n

    def training_step(self, data, batch_idx):
        (x_augmented, batch_augmented) = data.pos, data.batch

        if self.stn is not None:
            x_augmented = self.stn(x_augmented, batch_augmented)

        condition = self.sdm.inverse(x_augmented, batch_augmented)

        _, log_px, kinetic_energy, norm_jacobian = self.pdm.inverse(
            x_augmented,
            batch=batch_augmented,
            condition=condition,
            return_combined_dynamics=True,
        )

        nll = -scatter_sum(log_px, batch_augmented)
        n_samples_per_example = torch.bincount(batch_augmented)

        self.log(
            "train_loss",
            # MLE
            # -1/N sum_i log px_i
            (nll / n_samples_per_example).mean(),
            batch_size=data.num_graphs,
            sync_dist=True,
        )

        # Regularised MLE
        # 1/N sum_i -log px_i + \lambda_e KE + lambda_n FN
        loss = (
            nll
            + self.lambda_e * scatter_sum(kinetic_energy, batch_augmented)
            + self.lambda_n * scatter_sum(norm_jacobian, batch_augmented)
        ) / (n_samples_per_example * x_augmented.shape[1])

        loss = loss.mean()

        self.log(
            "regularised_train_loss",
            loss,
            batch_size=data.num_graphs,
            sync_dist=True,
        )

        return loss

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, data, batch_idx):
        (x_augmented, batch_augmented) = data.pos, data.batch

        if self.stn is not None:
            x_augmented = self.stn(x_augmented, batch_augmented)

        condition = self.sdm.inverse(x_augmented, batch_augmented)

        _, log_px, kinetic_energy, norm_jacobian = self.pdm.inverse(
            x_augmented,
            batch=batch_augmented,
            condition=condition,
            return_combined_dynamics=True,
        )

        nll = -scatter_sum(log_px, batch_augmented)
        n_samples_per_example = torch.bincount(batch_augmented)

        self.log(
            "valid_loss",
            # MLE
            # -1/N sum_i log px_i
            (nll / n_samples_per_example).mean(),
            batch_size=data.num_graphs,
            sync_dist=True,
        )

        # Regularised MLE
        # 1/N sum_i -log px_i + \lambda_e KE + lambda_n FN
        loss = (
            nll
            + self.lambda_e * scatter_sum(kinetic_energy, batch_augmented)
            + self.lambda_n * scatter_sum(norm_jacobian, batch_augmented)
        ) / (n_samples_per_example * x_augmented.shape[1])

        loss = loss.mean()

        self.log(
            "regularised_valid_loss",
            loss,
            batch_size=data.num_graphs,
            sync_dist=True,
        )

        return loss
