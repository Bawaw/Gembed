import torch
import pytorch_lightning as pl
from torch_scatter import scatter_mean, scatter_sum
from pytorch_lightning.utilities import grad_norm
import torch.nn as nn


class PointFlow(pl.LightningModule):
    """
    PointFlow is a PyTorch Lightning module that models a distribution using a shape distribution model (SDM)
    and a point distribution model (PDM).

    Args:
        sdm: The shape distribution model.
        pdm: The point distribution model.
    """

    def __init__(self, sdm, pdm):
        super().__init__()

        self.sdm = sdm
        self.pdm = pdm

    def pdm_inverse(self, **kwargs):
        """
        Compute the inverse of the point distribution model.
        $$
          f: x \rightarrow z
        $$

        Returns:
            torch.Tensor: The inverse of the point distribution model.
        """

        return self.pdm.inverse(**kwargs)

    def pdm_forward(self, **kwargs):
        """
        Compute the forward pass of the point distribution model.
        $$
          f: z \rightarrow x
        $$

        Returns:
            torch.Tensor: The output of the point distribution model.

        """
        return self.pdm.forward(**kwargs)

    def inverse(self, X, batch):
        """
        Compute the inverse of the shape distribution model for a given input and batch.
        $$
          f: X \rightarrow Z
        $$

        Args:
            X (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.

        Returns:
            torch.Tensor: The inverse of the shape distribution model.

        """
        condition = self.sdm.inverse(X, batch)
        return condition

    def log_prob(self, x, batch, condition=None):
        """
        Compute the log probability of the input `x` conditioned on the `condition`.
        $$
          p(x|condition)
        $$

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            condition (Optional[torch.Tensor]): The condition tensor. Default is None.

        Returns:
            torch.Tensor: The log probability of the input.

        """
        if condition is None:
            condition = self.inverse(x, batch)

        log_px = self.pdm.log_prob(x, batch, condition=condition)

        return log_px

    def log_prob_grad(self, x, batch, condition=None):
        """
        Compute the gradient of the log probability with respect to the input `x` under the condition `condition`.
        $$
          \nabla p(x|condition)
        $$

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

    def log_likelihood(self, x, batch):
        """
        Compute the log likelihood of the data.

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.

        Returns:
            torch.Tensor: The log likelihood of the data.

        """

        log_px = self.log_prob(x, batch)
        ll = scatter_mean(log_px, batch)
        return ll

    def training_step(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch_idx.batch
        nll = -self.log_likelihood(x, batch).mean()
        self.log("train_nll", nll, batch_size=train_batch.num_graphs)
        self.log("train_loss", nll, batch_size=train_batch.num_graphs)
        return nll

    def validation_step(self, valid_batch, batch_idx):
        x, batch = valid_batch.pos, valid_batch_idx.batch
        nll = -self.log_likelihood(x, batch).mean()
        self.log("valid_nll", nll, batch_size=valid_batch.num_graphs)
        self.log("valid_loss", nll, batch_size=valid_batch.num_graphs)
        return nll

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimiser


class PointFlowSTN(PointFlow):
    """
    PointFlowSTN is a subclass of PointFlow that incorporates a Spatial Transformer Network (STN) into the model.

    """

    def __init__(
        self,
        sdm,
        pdm,
        stn=None,
    ):
        super().__init__(sdm, pdm)

        self.stn = stn

    def pdm_inverse(
        self, x, batch=None, apply_stn=False, return_params=False, **kwargs
    ):
        """
        Compute the inverse of the point distribution model after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).

        Args:
            x (torch.Tensor): The input tensor.
            batch (Optional[torch.Tensor]): The batch tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The inverse of the point distribution model. If `apply_stn` is True, returns a tuple containing the inverse and the STN transformation parameters.

        """

        if self.stn is not None and apply_stn:
            x, params = self.stn(x, batch, return_params=True)

        result = super().pdm_inverse(x=x, batch=batch, **kwargs)

        if apply_stn and return_params:
            return result, params

        return result

    def pdm_forward(self, z, batch=None, apply_stn=False, stn_params=None, **kwargs):
        """
        Compute the forward pass of the point distribution model and optionally invert the spatial transformation.

        Args:
            z (torch.Tensor): The input tensor.
            batch (Optional[torch.Tensor]): The batch tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            stn_params: The parameters for the inverse STN transformation. Required when `apply_stn` is True.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The output of the point distribution model. If `apply_stn` is True, returns a tuple containing the output and the inverse STN transformation parameters.

        """

        x = super().pdm_forward(z=z, batch=batch, **kwargs)

        if self.stn is not None and apply_stn:
            assert stn_params is not None
            x = self.stn.inverse(x, batch, params=stn_params)

        return x

    def inverse(self, X, batch=None, apply_stn=False, return_params=False):
        """
        Compute the inverse of the shape distribution model for a given input and batch after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).

        Args:
            X (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The inverse of the shape distribution model. If `apply_stn` is True, returns a tuple containing the inverse and the STN transformation parameters.

        """

        if self.stn is not None and apply_stn:
            X, params = self.stn(X, batch, return_params=True)

        result = super().inverse(X=X, batch=batch)

        if return_params and apply_stn:
            if self.stn is not None:
                return result, params
            else:
                return result, None

        return result

    def log_prob(self, x, batch, condition=None, apply_stn=False, stn_params=None):
        """
        Compute the log probability of the input `x` conditioned on the `condition` after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).
        $$
          p(x|condition)
        $$

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            condition (Optional[torch.Tensor]): The condition tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.

        Returns:
            torch.Tensor: The log probability of the input.

        """
        if self.stn is not None and apply_stn:
            if stn_params is not None:
                x = self.stn(x, batch, stn_params)
            else:
                x = self.stn(x, batch)

        if condition is None:
            condition = self.inverse(x, batch)

        log_px = self.pdm.log_prob(x, batch, condition=condition)

        return log_px

    def log_prob_grad(self, x, batch, condition=None, apply_stn=False):
        """
        Compute the gradient of the log probability with respect to the input `x` under the condition `condition` after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).
        $$
          \nabla p(x|condition)
        $$

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            condition (Optional[torch.Tensor]): The condition tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.

        Returns:
            torch.Tensor: The gradient of the log probability with respect to `x`.

        """

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)

            if self.stn is not None and apply_stn:
                x = self.stn(x, batch)

            log_px = self.log_prob(x, batch, condition)

            # TODO: check that log_px is indeed a scalar
            nabla_log_px = torch.autograd.grad(log_px.sum(), x, create_graph=True)[0]

        return nabla_log_px

    def log_likelihood(self, data, batch_idx):
        """
        Compute the log likelihood of the data.

        Args:
            data (tg.Data): The input data.
            batch_idx: The indices of the current batch.

        Returns:
            torch.Tensor: The log likelihood of the data.

        """
        pos, batch = data.pos, batch_idx.batch

        log_px = self.log_prob(pos, batch)
        ll = scatter_mean(log_px, batch)
        return ll

    def training_step(self, train_batch, batch_idx):
        if self.stn is not None:
            train_batch.pos = self.stn(train_batch.pos, batch_idx.batch)

        return super().training_step(train_batch, batch_idx)

    def validation_step(self, valid_batch, batch_idx):
        if self.stn is not None:
            valid_batch.pos = self.stn(valid_batch.pos, batch_idx.batch)

        return super().training_step(valid_batch, batch_idx)

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)

        optim_dict = {"optimizer": optimiser}

        # plateau scheduler
        optim_dict["lr_scheduler"] = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                patience=int(1e4),
                verbose=True,
                min_lr=1e-7
                # optimiser, patience=int(1e3), verbose=True, min_lr=1e-7
            ),
            "monitor": "train_loss",
            "interval": "step",
        }

        return optim_dict


class RegularisedPointFlowSTN(PointFlowSTN):
    """TODO: Augmented Point Flow"""

    def __init__(self, lambda_e=1e-2, lambda_n=1e-2, lambda_m=0, **kwargs):
        super(RegularisedPointFlowSTN, self).__init__(**kwargs)

        self.lambda_e = lambda_e
        self.lambda_n = lambda_n
        self.lambda_m = lambda_m
        self.refinement_network = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
        from gembed.core.module.spectral import FourierFeatureMap

        self.refinement_network = nn.Sequential(
            FourierFeatureMap(3, 64, 1.0),
            nn.Linear(64, 64),
            # nn.Linear(3, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def manifold_loss(self, decode, z, eta=0.2):
        # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29
        bs = z.size(0)
        z_dim = z.size(1)

        # augment
        z_permuted = z[torch.randperm(bs)]
        alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha * z + (1 - alpha) * z_permuted

        # loss
        v = torch.randn(bs, z_dim).to(z_augmented)
        X, Jv = torch.autograd.functional.jvp(
            decode, z_augmented, v=v, create_graph=True
        )  # bs num_pts 3

        Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
        TrG = Jv_sq_norm.mean()

        # vTG(z)v - vTv c
        fm_loss = torch.mean(
            (Jv_sq_norm - (torch.sum(v ** 2, dim=1)) * TrG / z_dim) ** 2
        )
        return fm_loss.sum()

    # def compute_loss(self, train_batch, batch_idx, split, log=True):
    #     x, batch = train_batch.pos, train_batch.batch

    #     if self.stn is not None:
    #         x = self.stn(x, batch)

    #     # n_samples_per_example = torch.bincount(batch)
    #     n_samples_per_example = 512

    #     condition = self.inverse(x, batch=batch)
    #     # condition = torch.zeros(1, 64).to(x.device)

    #     # MAE
    #     if self.lambda_m == 0:
    #         RDM = 0.0
    #     else:
    #         n_rdm_samples = 100
    #         RDM = self.manifold_loss(
    #             lambda c: self.pdm.forward(
    #                 torch.randn(100, 3).to(x.device),
    #                 batch=None,
    #                 condition=c,
    #                 return_combined_dynamics=False,
    #             )[None],
    #             condition,
    #         )

    #     # AF
    #     x = x[:n_samples_per_example]
    #     batch = batch[:n_samples_per_example]
    #     _, log_px, kinetic_energy, norm_jacobian = self.pdm.inverse(
    #         x,
    #         batch=batch,
    #         condition=condition,
    #         include_combined_dynamics=True,
    #         include_log_density=True,
    #     )

    #     log_px, kinetic_energy, norm_jacobian = scatter_sum(
    #         torch.stack([log_px, kinetic_energy, norm_jacobian], -1), batch, dim=0
    #     ).T

    #     # AF AUG
    #     # x_aug = (1 / 3) * torch.randn_like(x)
    #     # _, log_px_aug, _, _ = self.pdm.inverse(
    #     #     x_aug,
    #     #     batch=batch,
    #     #     condition=condition,
    #     #     include_combined_dynamics=True,
    #     #     include_log_density=True,
    #     # )

    #     # log_px_aug = scatter_sum(log_px_aug, batch, dim=0)
    #     # NLL_aug = log_px_aug / n_samples_per_example
    #     # NLL_aug = 1e-2 * (NLL_aug / x.shape[1])
    #     # END AUG

    #     NLL = -(log_px / n_samples_per_example)
    #     KE = kinetic_energy / n_samples_per_example
    #     NJ = norm_jacobian / n_samples_per_example

    #     NLL = NLL / x.shape[1]
    #     KE = self.lambda_e * (KE / x.shape[1])
    #     NJ = self.lambda_n * (NJ / x.shape[1])
    #     RDM = self.lambda_m * RDM

    #     loss = NLL + KE + NJ + RDM  # + NLL_aug

    #     NLL, KE, NJ, loss = NLL.mean(), KE.mean(), NJ.mean(), loss.mean()

    #     if log:
    #         self.log(f"{split}_nll", NLL, batch_size=train_batch.num_graphs)
    #         self.log(f"{split}_ke", KE, batch_size=train_batch.num_graphs)
    #         self.log(f"{split}_nj", NJ, batch_size=train_batch.num_graphs)
    #         self.log(f"{split}_loss", loss, batch_size=train_batch.num_graphs)
    #         self.log(f"{split}_rdm", RDM, batch_size=train_batch.num_graphs)
    #         # self.log(f"{split}_nll_aug", NLL_aug, batch_size=train_batch.num_graphs)

    #     return loss

    def compute_loss(self, train_batch, batch_idx, split, log=True):

        x, batch = train_batch.pos, train_batch.batch

        if self.stn is not None:
            x = self.stn(x, batch)

        n_samples_per_example = torch.bincount(batch)
        # n_samples_per_example = 512

        condition = self.inverse(x, batch=batch)
        # condition = torch.zeros(1, 64).to(x.device)

        # MAE
        if self.lambda_m == 0:
            RDM = 0.0
        else:
            n_rdm_samples = 100
            RDM = self.manifold_loss(
                lambda c: self.pdm.forward(
                    torch.randn(100, 3).to(x.device),
                    batch=None,
                    condition=c,
                    return_combined_dynamics=False,
                )[None],
                condition,
            )

        # AF
        #x = x[:n_samples_per_example]
        #batch = batch[:n_samples_per_example]
        z, log_px, kinetic_energy, norm_jacobian = self.pdm.inverse(
            x,
            batch=batch,
            condition=condition,
            include_combined_dynamics=True,
            include_log_density=True,
        )

        # AF AUG
        # x_aug = (1 / 3) * torch.randn_like(x)
        # z_aug, log_px_aug, _, _ = self.pdm.inverse(
        #     x_aug,
        #     batch=batch,
        #     condition=condition,
        #     include_combined_dynamics=True,
        #     include_log_density=True,
        # )

        # xs = torch.cat([log_px, log_px_aug], 0).unsqueeze(1)
        # xs = torch.cat([z, z_aug], 0)
        # ys = torch.cat(
        #     [torch.ones_like(log_px), torch.zeros_like(log_px_aug)], 0
        # ).unsqueeze(1)
        # ys_pred = self.refinement_network(xs)

        # BCE = nn.functional.binary_cross_entropy(ys_pred, ys)

        # END AUG
        log_px, kinetic_energy, norm_jacobian = scatter_sum(
            torch.stack([log_px, kinetic_energy, norm_jacobian], -1), batch, dim=0
        ).T

        NLL = -(log_px / n_samples_per_example)
        KE = kinetic_energy / n_samples_per_example
        NJ = norm_jacobian / n_samples_per_example

        NLL = NLL / x.shape[1]
        KE = self.lambda_e * (KE / x.shape[1])
        NJ = self.lambda_n * (NJ / x.shape[1])
        RDM = self.lambda_m * RDM

        loss = NLL + KE + NJ + RDM  # + BCE  # + NLL_aug

        NLL, KE, NJ, loss = NLL.mean(), KE.mean(), NJ.mean(), loss.mean()

        if log:
            self.log(f"{split}_nll", NLL, batch_size=train_batch.num_graphs)
            self.log(f"{split}_ke", KE, batch_size=train_batch.num_graphs)
            self.log(f"{split}_nj", NJ, batch_size=train_batch.num_graphs)
            self.log(f"{split}_loss", loss, batch_size=train_batch.num_graphs)
            self.log(f"{split}_rdm", RDM, batch_size=train_batch.num_graphs)
            #self.log(f"{split}_bce", BCE, batch_size=train_batch.num_graphs)
            # self.log(f"{split}_nll_aug", NLL_aug, batch_size=train_batch.num_graphs)

        return loss

    # def on_before_optimizer_step(self, optimizer, X):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.pdm, norm_type=2)
    #     self.log_dict(norms)

    def training_step(self, train_batch, batch_idx):
        return self.compute_loss(train_batch, batch_idx, "train")

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, valid_batch, batch_idx):
        return self.compute_loss(train_batch, batch_idx, "valid")
