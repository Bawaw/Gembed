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

class PointScoreDiffusion(pl.LightningModule):
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
            self, X, batch=None, apply_stn=False, return_params=False, apply_ltn=False, apply_mtn=False,
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
        assert not (apply_mtn and apply_ltn), "Can not apply both ltn and mtn to same data."

        if self.stn is not None and apply_stn:
            X, params = self.stn(X, batch, return_params=True)

        Z = self.sdm.inverse(X, batch)

        if self.ltn is not None and apply_ltn:
            Z = self.ltn.inverse(x=Z, batch=torch.arange(Z.shape[0]))
        elif self.mtn is not None and apply_mtn:
            Z = self.mtn.inverse(X=Z)


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
        apply_mtn=False,
        stn_params=None,
        n_samples=int(8e4),
        **kwargs,
    ):
        assert not (apply_mtn and apply_ltn), "Can not apply both ltn and mtn to same data."

        if batch is not None:
            n_batch = batch.max() + 1
        else:
            n_batch = Z.shape[0]

        if self.ltn is not None and apply_ltn:
            Z = self.ltn.forward(z=Z, batch=torch.arange(Z.shape[0]))
        elif self.mtn is not None and apply_mtn:
            Z = self.mtn.forward(Z=Z)

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

    def _latent_metric_loss(self, func, z, eta=0.2, augment_z=True):
        # source: https://github.com/seungyeon-k/SMF-public/blob/e0a53e3b9ba48f4af091e6f11295c282cf535051/models/base_arch.py#L396C1-L420C29

        bs = z.size(0)
        z_dim = z.size(1)

        if augment_z:
            # interpolation based latent augmentation Note: requires
            assert bs > 1, "can not use Z augmentation if bs < 2"
            z_permuted = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha * z + (1 - alpha) * z_permuted

        else:
            z_augmented = z

        # loss
        v = torch.randn(bs, z_dim).to(z_augmented)

        X, Jv = torch.autograd.functional.jvp(
            func, z_augmented, v=v, create_graph=True
        )  # bs num_pts 3

        Jv_sq_norm = torch.einsum("nij,nij->n", Jv, Jv)
        c = Jv_sq_norm.mean() / z_dim

        # vTG(z)v - vTv c
        fm_loss = (Jv_sq_norm - (torch.sum(v ** 2, dim=1) * c)).pow(2)

        return fm_loss.mean()

    def _latent_metric_loss2(self, func, z, eta=0.2, augment_z=True):
        bs = z.size(0)
        z_dim = z.size(1)

        if augment_z:
            # interpolation based latent augmentation Note: requires
            assert bs > 1, "can not use Z augmentation if bs < 2"
            z_permuted = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha * z + (1 - alpha) * z_permuted

        else:
            z_augmented = z

        # loss
        v = torch.randn(bs, z_dim).to(z_augmented)

        Jv = torch.autograd.functional.jvp(
                func, z_augmented, v=v, create_graph=True)[1]
        TrG = torch.sum(Jv.view(bs, -1)**2, dim=1).mean()

        JTJv = (torch.autograd.functional.vjp(
            func, z_augmented, v=Jv, create_graph=True)[1]).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()

        return TrG2/TrG**2

    def latent_metric_loss(self, train_batch, batch_idx, n_samples=100, lambda_reg=1):
        # TODO: batch this
        x, batch = train_batch.pos, train_batch.batch

        # TODO: this is not the correct way to do it but we need to freeze batchnorm layers
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

        # REC LOSS
        C_rec = self.mtn.forward(Z_metric, time_steps=5)
        rec_loss = (C - C_rec).pow(2).mean(-1)
        rec_loss = rec_loss.mean()

        # METRIC LOSS
        z_template = 0.8 * torch.randn(
            n_samples, 3, generator=torch.Generator().manual_seed(42)
        ).repeat(batch.max()+1, 1).to(x)

        batch = (
            torch.concat([i * torch.ones(n_samples) for i in range(batch.max()+1)])
            .to(x)
            .long()
        )

        shape_generator = lambda z: self.forward(
            Z = self.mtn.forward(z),
            apply_pdm=True,
            time_steps=6,
            z=z_template,
            batch=batch,
        )[1].view(z.shape[0], -1, 3)

        # metric_loss = self._latent_metric_loss(shape_generator, Z_metric, augment_z=False)
        metric_loss = 1e-1 * self._latent_metric_loss2(shape_generator, Z_metric, augment_z=True)
        metric_loss = lambda_reg * metric_loss.mean()

        # TOTAL LOSS
        loss = rec_loss + metric_loss

        return loss, rec_loss, metric_loss

    def latent_metric_validation_loss(self, train_batch, batch_idx, n_cps=6):
        x, batch = train_batch.pos, train_batch.batch

        assert batch.max() == 1, "TODO: Batch this function."

        # TODO: this is not the correct way to do it
        # self.stn.eval()
        self.sdm.eval()
        self.pdm.eval()
        self.ltn.eval()

        if self.stn is not None:
            x = self.stn(x, batch)

        with torch.no_grad():
            C = self.sdm.inverse(x, batch)

            Z_metric = self.mtn.inverse(C)

            Z_metric = torch.lerp(
                input=Z_metric[:1],
                end=Z_metric[1:],
                weight=torch.linspace(0, 1, 6)[:, None].to(x.device),
            )

            C_interp = self.mtn.forward(Z_metric)
            reconstruction_error = (C - C_interp[[0, -1]]).pow(2).mean(-1)

            # replace start and end point by known point
            C_interp = torch.concat([C[:1], C_interp[1:-1], C[1:]])

            z_template = 0.8 * torch.randn(8000, 3).repeat(n_cps, 1).to(x)
            batch = torch.concat([i * torch.ones(8000) for i in range(n_cps)]).to(x).long()

            Xs = self.forward(
                Z=C_interp,
                apply_pdm=True,
                time_steps=10,
                n_samples=8000,  # 8000
                z=z_template,
                batch=batch,
            )[1].view(n_cps, -1, 3)

            delta_t = 1 / (n_cps - 1)
            geodesic_energy = 0.5*(Xs[:-1] - Xs[1:]).pow(2).sum(-1).mean(-1).div(delta_t).sum()

            # from gembed.vis import plot_objects
            # plot_objects(
            #     (Xs[0].cpu(), None),
            #     (Xs[1].cpu(), None),
            #     (Xs[2].cpu(), None),
            #     (Xs[3].cpu(), None),
            #     (Xs[4].cpu(), None),
            #     (Xs[5].cpu(), None),
            # )

        return geodesic_energy, reconstruction_error

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

            self.log("valid_point_ll", ll.mean(), batch_size=valid_batch.num_graphs)

        elif self.phase == Phase.TRAIN_LATENT_DIFFUSION:
            # TODO:
            return None
            # loss = self.latent_diffusion_loss(train_batch, batch_idx)
            # self.log("train_latent_ll", loss, batch_size=train_batch.num_graphs)

        elif self.phase == Phase.TRAIN_METRIC_TRANSFORMER:
            ge, re = self.latent_metric_validation_loss(valid_batch, batch_idx)
            self.log("valid_metric_geodesic", ge.mean(), batch_size=valid_batch.num_graphs)
            self.log("valid_metric_reconstruction", re.mean(), batch_size=valid_batch.num_graphs)
