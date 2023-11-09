import torch
import pytorch_lightning as pl
from torch_scatter import scatter_mean, scatter_sum
from pytorch_lightning.utilities import grad_norm
import functools
import numpy as np
from enum import Enum


class Phase(Enum):
    TRAIN_POINT_DIFFUSION = 1
    TRAIN_LATENT_DIFFUSION = 2
    TRAIN_METRIC_TRANSFORMER = 3
    EVAL = 4


# class PointScoreDiffusion(pl.LightningModule):
#     """
#     PointFlow is a PyTorch Lightning module that models a distribution using a shape distribution model (SDM)
#     and a point distribution model (PDM).

#     Args:
#         sdm: The shape distribution model.
#         pdm: The point distribution model.
#     """

#     def __init__(self, sdm, pdm):
#         super().__init__()

#         self.sdm = sdm
#         self.pdm = pdm

#     def pdm_inverse(self, **kwargs):
#         """
#         Compute the inverse of the point distribution model.
#         $$
#           f: x \rightarrow z
#         $$

#         Returns:
#             torch.Tensor: The inverse of the point distribution model.
#         """

#         return self.pdm.inverse(**kwargs)

#     def pdm_forward(self, **kwargs):
#         """
#         Compute the forward pass of the point distribution model.
#         $$
#           f: z \rightarrow x
#         $$

#         Returns:
#             torch.Tensor: The output of the point distribution model.

#         """
#         return self.pdm.forward(**kwargs)

#     def inverse(self, X, batch):
#         """
#         Compute the inverse of the shape distribution model for a given input and batch.
#         $$
#           f: X \rightarrow Z
#         $$

#         Args:
#             X (torch.Tensor): The input tensor.
#             batch (torch.Tensor): The batch tensor.

#         Returns:
#             torch.Tensor: The inverse of the shape distribution model.

#         """
#         condition = self.sdm.inverse(X, batch)
#         return condition

#     def log_prob(self, x, batch, condition=None):
#         """
#         Compute the log probability of the input `x` conditioned on the `condition`.
#         $$
#           p(x|condition)
#         $$

#         Args:
#             x (torch.Tensor): The input tensor.
#             batch (torch.Tensor): The batch tensor.
#             condition (Optional[torch.Tensor]): The condition tensor. Default is None.

#         Returns:
#             torch.Tensor: The log probability of the input.

#         """
#         if condition is None:
#             condition = self.inverse(x, batch)

#         log_px = self.pdm.log_prob(x, batch, condition=condition)

#         return log_px

#     def log_prob_grad(self, x, batch, condition=None):
#         """
#         Compute the gradient of the log probability with respect to the input `x` under the condition `condition`.
#         $$
#           \nabla p(x|condition)
#         $$

#         Args:
#             x (torch.Tensor): The input tensor.
#             batch (torch.Tensor): The batch tensor.
#             condition (Optional[torch.Tensor]): The condition tensor. Default is None.

#         Returns:
#             torch.Tensor: The gradient of the log probability with respect to `x`.

#         """

#         with torch.set_grad_enabled(True):
#             x = x.requires_grad_(True)
#             log_px = self.log_prob(x, batch, condition)

#             # TODO: check that log_px is indeed a scalar
#             nabla_log_px = torch.autograd.grad(log_px.sum(), x, create_graph=True)[0]

#         return nabla_log_px

#     def log_likelihood(self, x, batch, **kwargs):
#         """
#         Compute the log likelihood of the data.

#         Args:
#             x (torch.Tensor): The input tensor.
#             batch (torch.Tensor): The batch tensor.

#         Returns:
#             torch.Tensor: The log likelihood of the data.

#         """

#         log_px = self.log_prob(x, batch, **kwargs)
#         ll = scatter_mean(log_px, batch)
#         return ll


class PointScoreDiffusionSTN(pl.LightningModule):
    """
    PointFlowSTN is a subclass of PointFlow that incorporates a Spatial Transformer Network (STN) into the model.

    """

    def __init__(
        self, sdm, pdm, phase=None, stn=None, ltn=None, mtn=None, lambda_kld=0
    ):
        super().__init__()

        # modules
        self.sdm = sdm
        self.pdm = pdm
        self.stn = stn
        self.ltn = ltn
        self.mtn = mtn

        # variables
        self.lambda_kld = lambda_kld
        if phase is None:
            self.set_phase(Phase.TRAIN_POINT_DIFFUSION)

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

        z = self.pdm.inverse(x=x, batch=batch, **kwargs)

        if apply_stn and return_params:
            return z, params

        return z

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

        x = self.pdm.forward(z=z, batch=batch, **kwargs)

        if self.stn is not None and apply_stn:
            assert stn_params is not None
            x = self.stn.inverse(x, batch, params=stn_params)

        return x

    def inverse(
        self, X, batch=None, apply_stn=False, return_params=False, apply_ltn=False
    ):
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

        Z = self.sdm.inverse(X, batch)

        if self.ltn is not None and apply_ltn:
            Z = self.ltn.inverse(x=Z, batch=torch.arange(Z.shape[0]))

        if return_params and apply_stn:
            if self.stn is not None:
                return Z, params
            else:
                return Z, None
        return Z

    def forward(
        self,
        Z,
        z=None,
        batch=None,
        apply_stn=False,
        apply_pdm=False,
        apply_ltn=False,
        stn_params=None,
        n_samples=int(8e4),
        **kwargs,
    ):
        if batch is not None:
            n_batch = batch.max() + 1
        else:
            n_batch = Z.shape[0]

        if self.ltn is not None and apply_ltn:
            Z = self.ltn.forward(z=Z, batch=torch.arange(Z.shape[0]))

        if not apply_pdm:
            return Z

        if z is None:
            z = self.pdm.sample_base(n_batch * int(n_samples))

            if batch is None:
                batch = torch.concat(
                    [i * torch.ones(n_samples) for i in range(n_batch)]
                ).long()

        X_rec = self.pdm_forward(
            z=z,
            batch=batch,
            apply_stn=apply_stn,
            stn_params=stn_params,
            condition=Z,
            **kwargs,
        )

        return Z, X_rec

    def log_prob(
        self, x, batch, condition=None, apply_stn=False, stn_params=None, **kwargs
    ):
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

        log_px = self.pdm.log_prob(x, batch, condition=condition, **kwargs)

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
        raise NotImplementedError

    def log_likelihood(self, x, batch, **kwargs):
        log_px = self.log_prob(x, batch, **kwargs)
        ll = scatter_mean(log_px, batch)
        return ll

    def set_phase(self, phase):
        self.phase = phase

        if phase == Phase.TRAIN_POINT_DIFFUSION:
            if self.ltn is not None:
                self.ltn.freeze(True)
            if self.stn is not None:
                self.stn.freeze(False)
            if self.mtn is not None:
                self.mtn.freeze(True)
            self.sdm.freeze(False)
            self.pdm.freeze(False)

        elif phase == Phase.TRAIN_LATENT_DIFFUSION:
            if self.ltn is not None:
                self.ltn.freeze(False)
            if self.stn is not None:
                self.stn.freeze(True)
            if self.mtn is not None:
                self.mtn.freeze(True)
            self.sdm.freeze(True)
            self.pdm.freeze(True)

        elif phase == Phase.TRAIN_LATENT_DIFFUSION:
            if self.ltn is not None:
                self.ltn.freeze(False)
            if self.stn is not None:
                self.stn.freeze(True)
            if self.mtn is not None:
                self.mtn.freeze(True)
            self.sdm.freeze(True)
            self.pdm.freeze(True)

        elif phase == Phase.TRAIN_METRIC_TRANSFORMER:
            if self.ltn is not None:
                self.ltn.freeze(True)
            if self.stn is not None:
                self.stn.freeze(True)
            if self.mtn is not None:
                self.mtn.freeze(False)
            self.sdm.freeze(True)
            self.pdm.freeze(True)

        else:
            if self.ltn is not None:
                self.ltn.freeze(True)
            if self.stn is not None:
                self.stn.freeze(True)
            if self.mtn is not None:
                self.mtn.freeze(True)
            self.sdm.freeze(True)
            self.pdm.freeze(True)

    def configure_optimizers(self):
        if self.phase == Phase.TRAIN_POINT_DIFFUSION and self.stn is not None:
            optimiser = torch.optim.Adam(
                [
                    {"params": self.stn.parameters(), "weight_decay": 1e-6},
                    {"params": self.pdm.parameters()},
                    {"params": self.sdm.parameters()},
                ],
                lr=1e-3,
                weight_decay=0.0,
            )
        elif self.phase == Phase.TRAIN_METRIC_TRANSFORMER:
            optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimiser

    # def on_before_optimizer_step(self, optimizer, X):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.pdm, norm_type=2)
    #     self.log_dict(norms)

    def latent_diffusion_loss(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch.batch
        batch_size = batch.max() + 1

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            Z_mean, Z_log_var = self.sdm.get_params(x, batch=batch)
            Z_std = torch.exp(0.5 * Z_log_var)
            condition = Z_mean + Z_std * torch.randn_like(Z_mean)
        else:
            condition = self.sdm.inverse(x, batch)

        condition_batch = torch.arange(0, condition.shape[0])

        # t ~ U(0, 1)
        t = torch.rand((batch_size, 1), device=x.device).to(x.device)
        # t = 1.0 * torch.ones((batch_size, 1), device=x.device).to(x.device)

        # p_t(x) = N(μ, σ)
        mean, std = self.ltn.marginal_prob_params(condition, t, condition_batch)

        # # \tilde{x} ~ p_t(x)
        eps = torch.randn_like(condition)
        condition_tilde = mean + std * eps

        # # s(\tilde{x}(t), t) = - std * eps
        score = self.ltn.score(x=condition_tilde, t=t, batch=condition_batch)

        loss = 0.5 * (score * std + eps).pow(2).sum(-1)

        # # aggregate over batches
        loss = loss.mean()

        return loss

    def point_diffusion_loss(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch.batch
        batch_size = batch.max() + 1

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            Z_mean, Z_log_var = self.sdm.get_params(x, batch=batch)
            Z_std = torch.exp(0.5 * Z_log_var)
            condition = Z_mean + Z_std * torch.randn_like(Z_mean)
            KLD = -0.5 * torch.sum(1 + Z_log_var - Z_mean.pow(2) - Z_log_var.exp())
        else:
            condition = self.sdm.inverse(x, batch)

        # t ~ U(0, 1)
        t = torch.rand((batch_size, 1), device=x.device).to(x.device)
        # t = torch.rand(batch_size, device=x.device) * (1.0 - eta) + eta

        # p_t(x) = N(μ, σ)
        mean, std = self.pdm.marginal_prob_params(x, t, batch, condition)

        # \tilde{x} ~ p_t(x)
        eps = torch.randn_like(x)
        x_tilde = mean + std * eps

        # s(\tilde{x}(t), t) = - std * eps
        score = self.pdm.score(x=x_tilde, t=t, batch=batch, condition=condition)

        loss = 0.5 * (score * std + eps).pow(2).sum(-1)

        # aggregate over batches
        loss = loss.mean()

        if self.lambda_kld > 0:
            loss += self.lambda_kld * KLD

        return loss

    def _latent_metric_loss_discrete(self, func, z, eta=0.2):
        bs = z.size(0)
        z_dim = z.size(1)

        # α ∈ [-η, +η]
        # z_permuted = z[[1, 0]]
        # alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
        # Z_augmented = alpha * z + (1 - alpha) * z_permuted
        Z_augmented = z

        # Synthesise shapes
        Xs = func(Z_augmented)

        # minimises distance between shapes
        loss = (Xs[:-1] - Xs[1:]).pow(2).mean([1, 2]).mean()

        # from gembed.vis import plot_objects

        # plot_objects(
        #     (Xs[0].cpu(), None),
        #     (Xs[1].cpu(), None),
        # )

        return loss.mean()

    def _latent_metric_loss_continuous(self, func, z, eta=0.2, augment_z=True):
        # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29

        bs = z.size(0)
        z_dim = z.size(1)

        # augment
        if augment_z:
            z_permuted = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha * z + (1 - alpha) * z_permuted
        else:
            z_augmented = z

        # loss
        v = torch.randn(bs, z_dim).to(z_augmented)
        # v = (
        #     torch.from_numpy(np.random.RandomState(42).randn(bs, z_dim))
        #     .float()
        #     .to(z.device)
        # )

        X, Jv = torch.autograd.functional.jvp(
            func, z_augmented, v=v, create_graph=True
        )  # bs num_pts 3

        Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
        c = Jv_sq_norm.mean() / z_dim
        # c = self.mtn.constant
        # c = 1

        # vTG(z)v - vTv c
        fm_loss = (Jv_sq_norm - (torch.sum(v ** 2, dim=1) * c)).pow(2)

        return fm_loss.mean()

    def _latent_metric_loss_exact(self, func, z, eta=0.2):
        # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29

        bs = z.size(0)
        z_dim = z.size(1)
        foo = lambda z: func(z[None]).view(-1)

        c = self.mtn.constant

        J = torch.autograd.functional.jacobian(foo, z[0], strict=True)
        JTJ = torch.einsum("nij,nik->njk", J[None], J[None])

        loss = (JTJ - c * torch.eye(z_dim).to(z)).norm(dim=[1, 2], p="fro")

        return loss.mean()

    # def _latent_metric_loss(self, decode, z, eta=0.2):
    #     # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29

    #     bs = z.size(0)
    #     z_dim = z.size(1)

    #     # augment
    #     z_permuted = z[torch.randperm(bs)]
    #     alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
    #     z_augmented = alpha * z + (1 - alpha) * z_permuted

    #     # loss
    #     v = torch.randn(bs, z_dim).to(z_augmented)

    #     X, Jv = torch.autograd.functional.jvp(
    #         decode, z_augmented, v=v, create_graph=True
    #     )  # bs num_pts 3

    #     Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
    #     TrG = Jv_sq_norm.mean()

    #     # vTG(z)v - vTv c
    #     fm_loss = (Jv_sq_norm - (torch.sum(v ** 2, dim=1) * (TrG / z_dim))).pow(2)

    #     fm_loss = fm_loss.clamp(max=1)

    #     return fm_loss.mean()

    def _latent_metric_loss_continuous_rdm(self, func, z, eta=0.2, create_graph=True):
        """
        func: decoder that maps "latent value z" to "data", where z.size() == (batch_size, latent_dim)
        """
        bs = len(z)
        # z_perm = z[torch.randperm(bs)]
        z_perm = z[[1, 0]]
        alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
        z_augmented = alpha * z + (1 - alpha) * z_perm
        v = torch.randn(z.size()).to(z)
        Jv = torch.autograd.functional.jvp(
            func, z_augmented, v=v, create_graph=create_graph
        )[1]
        TrG = torch.sum(Jv.view(bs, -1) ** 2, dim=1).mean()
        JTJv = (
            torch.autograd.functional.vjp(
                func, z_augmented, v=Jv, create_graph=create_graph
            )[1]
        ).view(bs, -1)
        TrG2 = torch.sum(JTJv ** 2, dim=1).mean()
        return TrG2 / TrG ** 2

    def _latent_metric_loss(self, decode, Zs):
        # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29

        bs = Zs.size(0)
        Zs_dim = Zs.size(1)

        # loss
        v = torch.randn(bs, Zs_dim).to(Zs)
        # Zs = Zs.repeat(100, 1)
        # v = (
        #     torch.from_numpy(np.random.RandomState(42).randn(bs, Zs_dim))
        #     .float()
        #     .to(Zs.device)
        # )

        # X, Jv = torch.autograd.functional.jvp(
        #     decode, Zs, v=v, create_graph=True
        # )  # bs num_pts 3

        # Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
        # TrG = Jv_sq_norm.mean()
        #
        # loss = (
        #     (torch.sum(v ** 2, dim=1)) -
        #     (decode(Zs) - decode(Zs+v)).pow(2).sum(-1)
        # ).pow(2)

        # vTG(Zs)v - vTv c
        # c = self.mtn.constant(Zs)
        # c = Jv_sq_norm.mean()/Zs_dim
        # loss = (Jv_sq_norm - (c * torch.sum(v ** 2, dim=1))).pow(2)
        # loss = (Jv_sq_norm - (c * torch.sum(v ** 2, dim=1))).pow(2)

        # from gembed.vis import plot_objects
        # Xs = decode(Zs)
        # plot_objects((Xs[0].cpu(), None))

        Xs = decode(Zs)
        loss = (Xs[:-1] - Xs[1:]).pow(2).mean([1, 2])

        return loss

    # def latent_metric_loss(self, train_batch, batch_idx):
    #     x, batch = train_batch.pos, train_batch.batch

    #     # TODO: this is not the correct way to do it
    #     # self.stn.eval()
    #     self.sdm.eval()
    #     self.pdm.eval()
    #     self.ltn.eval()

    #     if self.stn is not None:
    #         x = self.stn(x, batch)

    #     if self.lambda_kld > 0:
    #         C_mean, C_log_var = self.sdm.get_params(x, batch=batch)
    #         C_std = torch.exp(0.5 * C_log_var)
    #         C = C_mean + C_std * torch.randn_like(C_mean)
    #     else:
    #         C = self.sdm.inverse(x, batch)

    #     Z_metric = self.mtn.inverse(C)
    #     # C_rec = self.mtn.forward(z)

    #     # t ~ U(0, 1)
    #     t = torch.rand((batch.max() + 1, 1), device=x.device).to(x.device)
    #     mean, std = self.pdm.marginal_prob_params(x, t, batch, C)

    #     # \tilde{x} ~ p_t(x)
    #     eps = torch.randn_like(x)
    #     x_tilde = mean + std * eps

    #     func = lambda condition: self.pdm.inverse_drift_coefficient(
    #         x_tilde, t, batch, self.mtn.forward(condition)
    #     ).view(batch.max() + 1, -1, 3)

    #     # loss = self._latent_metric_loss_discrete(func, Z_metric)
    #     # loss = self._latent_metric_loss_exact(func, Z_metric)
    #     loss = self._latent_metric_loss_continuous(func, Z_metric)
    #     # loss = self._latent_metric_loss_continuous_rdm(func, Z_metric)
    #     return loss.mean()

    #     # C_rec = self.mtn.forward(Z_metric)
    #     # rec = (C - C_rec).pow(2).mean(-1)
    #     # return rec.mean() + loss.mean()

    def latent_metric_loss(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch.batch

        # TODO: this is not the correct way to do it
        # self.stn.eval()
        self.sdm.eval()
        self.pdm.eval()
        self.ltn.eval()

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            C_mean, C_log_var = self.sdm.get_params(x, batch=batch)
            C_std = torch.exp(0.5 * C_log_var)
            C = C_mean + C_std * torch.randn_like(C_mean)
        else:
            C = self.sdm.inverse(x, batch)

        Z_metric = self.mtn.inverse(C, time_steps=5)

        # z_template = 0.8 * torch.randn(1000, 3).repeat(batch.max() + 1, 1).to(x)
        z_template = 0.8 * torch.randn(
            100, 3, generator=torch.Generator().manual_seed(42)
        ).repeat(batch.max() + 1, 1).to(x)
        batch = (
            torch.concat([i * torch.ones(100) for i in range(batch.max() + 1)])
            .to(x)
            .long()
        )

        func = lambda z: self.forward(
            self.mtn.forward(z, time_steps=5),
            apply_pdm=True,
            time_steps=6,
            z=z_template,
            batch=batch,
        )[1].view(z.shape[0], -1, 3)

        # loss = self._latent_metric_loss_discrete(func, Z_metric)
        # loss = self._latent_metric_loss_exact(func, Z_metric)
        loss = self._latent_metric_loss_continuous(func, Z_metric, augment_z=False)
        # loss = self._latent_metric_loss_continuous_rdm(func, Z_metric)
        # return loss.mean()

        C_rec = self.mtn.forward(Z_metric, time_steps=5)
        rec = (C - C_rec).pow(2).mean(-1)

        REC = rec.mean()
        REG = loss.mean()

        lambda_reg = int(1e4)
        return REC + lambda_reg*REG, REC, lambda_reg*REG

    def latent_metric_validation_loss(self, train_batch, batch_idx):
        x, batch = train_batch.pos, train_batch.batch

        # TODO: this is not the correct way to do it
        # self.stn.eval()
        self.sdm.eval()
        self.pdm.eval()
        self.ltn.eval()

        if self.stn is not None:
            x = self.stn(x, batch)

            # if self.lambda_kld > 0:
            #     C_mean, C_log_var = self.sdm.get_params(x, batch=batch)
            #     C_std = torch.exp(0.5 * C_log_var)
            #     C = C_mean + C_std * torch.randn_like(C_mean)
            # else:
        with torch.no_grad():
            C = self.sdm.inverse(x, batch)

            Z_metric = self.mtn.inverse(C)

            Z_metric = torch.lerp(
                input=Z_metric[:1],
                end=Z_metric[1:],
                weight=torch.linspace(0, 1, 6)[:, None].to(x.device),
            )

            z_template = 0.8 * torch.randn(8000, 3).repeat(6, 1).to(x)
            batch = torch.concat([i * torch.ones(8000) for i in range(6)]).to(x).long()

            Xs = self.forward(
                self.mtn.forward(Z_metric),
                apply_pdm=True,
                time_steps=10,
                n_samples=8000,  # 8000
                z=z_template,
                batch=batch,
            )[1].view(6, -1, 3)

            loss = (Xs[:-1] - Xs[1:]).pow(2).sum([1, 2])

            # from gembed.vis import plot_objects

            # plot_objects(
            #     (Xs[0].cpu(), None),
            #     (Xs[1].cpu(), None),
            #     (Xs[2].cpu(), None),
            #     (Xs[3].cpu(), None),
            #     (Xs[4].cpu(), None),
            #     (Xs[5].cpu(), None),
            # )

        return loss

    def training_step(self, train_batch, batch_idx):
        if self.phase == Phase.TRAIN_POINT_DIFFUSION:
            loss = self.point_diffusion_loss(train_batch, batch_idx)
            self.log("train_point_loss", loss, batch_size=train_batch.num_graphs)

        elif self.phase == Phase.TRAIN_LATENT_DIFFUSION:
            loss = self.latent_diffusion_loss(train_batch, batch_idx)
            self.log("train_latent_loss", loss, batch_size=train_batch.num_graphs)

        elif self.phase == Phase.TRAIN_METRIC_TRANSFORMER:
            loss, rec, reg = self.latent_metric_loss(train_batch, batch_idx)
            self.log("train_metric_loss", loss, batch_size=train_batch.num_graphs)
            self.log("train_metric_reg", reg, batch_size=train_batch.num_graphs)
            self.log("train_metric_rec", rec, batch_size=train_batch.num_graphs)

        else:
            assert False, f"Invalid phase {self.phase} for training"

        return loss

    # def validation_step(self, valid_batch, batch_idx):
    def validation_step(self, valid_batch, batch_idx):
        if self.phase == Phase.TRAIN_POINT_DIFFUSION:
            x, batch = valid_batch.pos, valid_batch.batch

            with torch.no_grad():
                ll = self.log_likelihood(x, batch, time_steps=10, apply_stn=True)

            self.log("valid_ll", ll.mean(), batch_size=valid_batch.num_graphs)

        elif self.phase == Phase.TRAIN_METRIC_TRANSFORMER:
            ml = self.latent_metric_validation_loss(valid_batch, batch_idx)
            self.log("valid_metric_loss", ml.mean(), batch_size=valid_batch.num_graphs)
