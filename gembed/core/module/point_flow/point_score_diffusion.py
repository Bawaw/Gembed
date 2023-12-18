from copy import deepcopy
import functools
from enum import Enum

import numpy as np
import lightning as pl
import torch
from torch_scatter import scatter_mean, scatter_sum

from gembed.core.module import InvertibleModule


class Phase(Enum):
    TRAIN_POINT_DIFFUSION = 1
    TRAIN_LATENT_DIFFUSION = 2
    TRAIN_METRIC_TRANSFORMER = 3
    EVAL = 4


class PointScoreDiffusion(pl.LightningModule):
    """ This PyTorch Lightning module implements a model with 5 components:
        1. Shape Distribution Model (SDM):
        - Maps shape to latent representation.
        - Modelled as a GDL Variational encoder.

        2. Point Distribution Model (PDM):
        - Models point distribution conditioned on shape latent representation.
        - Modelled as a continuous diffusion model.

        3. Spatial Transformer Network (STN):
        - Superimposes all input shapes.
        - Modelled as a GDL regression model.

        4. Latent Transformer Network (LTN):
        - Maps latent shape distribution to a normal distribution.
        - Modelled as a regularized encoder-decoder.

        5. Metric Transformer Network (MTN):
        - Maps latent shape space to a metric space.
        - Modelled as a diffusion model.
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
        """Compute the inverse of the point distribution model after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).

        Args:
            x (torch.Tensor): The input tensor.
            batch (Optional[torch.Tensor]): The batch tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            return_params (bool): Whether to return the transformation parameters. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The inverse of the point distribution model. If `apply_stn` and `return_params` is True, returns a tuple containing the inverse and the STN transformation parameters.

        """

        if self.stn is not None and apply_stn:
            x, params = self.stn(x, batch, return_params=True)

        z = self.pdm.inverse(x=x, batch=batch, **kwargs)

        if apply_stn and return_params:
            return z, params

        return z

    def pdm_forward(self, z, batch=None, apply_stn=False, stn_params=None, **kwargs):
        """Compute the forward pass of the point distribution model and optionally invert the spatial transformation.

        Args:
            z (torch.Tensor): The input tensor.
            batch (Optional[torch.Tensor]): The batch tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            stn_params: The parameters for the inverse STN transformation. Required when `apply_stn` is True.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The output of the point distribution model.

        """

        x = self.pdm.forward(z=z, batch=batch, **kwargs)

        if self.stn is not None and apply_stn:
            assert stn_params is not None
            x = self.stn.inverse(x, batch, params=stn_params)

        return x

    def inverse(
        self,
        X,
        batch=None,
        apply_stn=False,
        return_params=False,
        apply_ltn=False,
        apply_mtn=False,
    ):
        """Compute the inverse of the shape distribution model for a given input and batch after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).

        Args:
            X (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            return_params (bool): Whether to return the transformation parameters. Default is False.
            apply_ltn (bool): Whether to apply the latent transformer network and return its output. Default is False.
            apply_mtn (bool): Whether to apply the metric transformer network and return its output. Default is False.

        Notes:
            `apply_ltn` and `apply_mtn` can not both be set to True.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The inverse of the shape distribution model. If `apply_stn` is True, returns a tuple containing the inverse and the STN transformation parameters.

        """
        assert not (
            apply_mtn and apply_ltn
        ), "Can not apply both ltn and mtn to same data."

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
        """The forward of the LTN, MTN or SDM. This method can be used to synthesise new shapes.

        Args:
            Z (torch.Tensor): Input tensor representing the representation (either in metricised, normalised or latent space).
            z (torch.Tensor, optional): Tensor representing the latent point representation. Default is None.
            batch (torch.Tensor, optional): Batch tensor. Default is None.
            apply_stn (bool, optional): Flag to apply the Spatial Transformer Network (STN) transformation. Default is False.
            apply_pdm (bool, optional): Flag to apply the Point Distribution Model (PDM). Default is False.
            apply_ltn (bool, optional): Flag to apply the Latent Transformer Network (LTN) transformation. Default is False.
            apply_mtn (bool, optional): Flag to apply the Metric Transformer Network (MTN) transformation. Default is False.
            stn_params (Any, optional): Parameters for the STN transformation. Default is None.
            n_samples (int, optional): Number of samples for shape synthesis (only used if `z` is None). Default is int(8e4).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: If apply_pdm is False, returns the latent space Z. Otherwise, returns a tuple containing the latent space Z and the reconstructed output X_rec.

        Notes:
            The method handles optional application of transformations (STN, LTN, MTN) and point diffusion based on the provided flags.
            If apply_pdm is True, the method synthesises a shape using the Point Distribution Model (PDM) and provided conditions.
        """
        assert not (
            apply_mtn and apply_ltn
        ), "Can not apply both ltn and mtn to same data."

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

        X = self.pdm_forward(
            z=z,
            batch=batch,
            apply_stn=apply_stn,
            stn_params=stn_params,
            condition=Z,
            **kwargs,
        )

        return Z, X

    def log_prob(
        self, x, batch, condition=None, apply_stn=False, stn_params=None, **kwargs
    ):
        """Compute the log density of the input `x` conditioned on the `condition` after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).
        $$
          p(x|condition)
        $$

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            condition (Optional[torch.Tensor]): The condition tensor. Default is None.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            stn_params (Optional[torch.Tensor]): The transformation parameters to apply. Default None.

        Note:
            If `apply_stn` and `stn_params` is None, the STN will predict the transformation parameters.
            If `condition` is None, the SDM will predict the condition.

        Returns:
            torch.Tensor: The log density of the input.

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

    def log_prob_Z(self, x, batch, apply_stn=False, stn_params=None, **kwargs):
        """Compute the log density of the input point cloud's representation $Z$ after optionally transforming the data to canonical coordinates using the Spatial Transformer Network (STN).
        $$
          p(condition)
        $$

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): The batch tensor.
            apply_stn (bool): Whether to apply the STN transformation. Default is False.
            stn_params (Optional[torch.Tensor]): The transformation parameters to apply. Default None.

        Note:
            If `apply_stn` and `stn_params` is None, the STN will predict the transformation parameters.

        Returns:
            torch.Tensor: The log density of the input.

        """
        if self.stn is not None and apply_stn:
            if stn_params is not None:
                x = self.stn(x, batch, stn_params)
            else:
                x = self.stn(x, batch)

        Z = self.inverse(x, batch)

        Z_batch = torch.Tensor([i for i in range(Z.shape[0])]).long()
        log_pX = self.ltn.log_prob(Z, Z_batch, condition=None, **kwargs)

        return log_pX

    def log_likelihood(self, x, batch, **kwargs):
        """Computes the average log density per shape (determined by batch).

        Args:
            x (torch.Tensor): Input tensor.
            batch (torch.Tensor): Batch tensor.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: Average log density of the input data.

        """
        log_px = self.log_prob(x, batch, **kwargs)
        ll = scatter_mean(log_px, batch)
        return ll

    def set_phase(self, phase):
        """This method sets the training phase of the PointScoreDiffusion model, controlling the freezing and unfreezing of model components based on the specified phase.

        Args:
            phase (Phase): Training phase enum value.

        Returns:
            None
        """

        self.phase = phase

        if phase == Phase.TRAIN_POINT_DIFFUSION:
            if self.ltn is not None:
                self.ltn.freeze()
            if self.stn is not None:
                self.stn.unfreeze()
            if self.mtn is not None:
                self.mtn.freeze()
            self.sdm.unfreeze()
            self.pdm.unfreeze()

        elif phase == Phase.TRAIN_LATENT_DIFFUSION:
            if self.ltn is not None:
                self.ltn.unfreeze()
            if self.stn is not None:
                self.stn.freeze()
            if self.mtn is not None:
                self.mtn.freeze()
            self.sdm.freeze()
            self.pdm.freeze()

        elif phase == Phase.TRAIN_METRIC_TRANSFORMER:
            if self.ltn is not None:
                self.ltn.freeze()
            if self.stn is not None:
                self.stn.freeze()
            if self.mtn is not None:
                self.mtn.unfreeze()
            self.sdm.freeze()
            self.pdm.freeze()

        else:
            if self.ltn is not None:
                self.ltn.unfreeze()
            if self.stn is not None:
                self.stn.unfreeze()
            if self.mtn is not None:
                self.mtn.unfreeze()
            self.sdm.unfreeze()
            self.pdm.unfreeze()
            self.eval()

    def on_train_start(self):
        """Calls `set_phase` with the current `self.phase` on the start of training to ensure that the 
        freezing and unfreezing of layers is working as expected. 

        Note: This function is essential because PyTorch Lightning calls train() on all modules at the 
        start of training.  This changes the behaviour of e.g. BatchNorm, and complicates training.
        """
        super().on_train_start()
        
        self.set_phase(self.phase)


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

    # TRAINING STEP
    def point_diffusion_loss(self, train_batch, batch_idx):
        """This method computes the loss for the main training phase of the PointScoreDiffusion model (training of pdm, sdm, stn).

        Args:
            train_batch (torch_geometric.data.Data): Training batch containing input data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the point diffusion phase.
        
        source: https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py
        """

        x, batch = train_batch.pos, train_batch.batch
        batch_size = batch.max() + 1

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            Z_mean, Z_log_var = self.sdm.get_params(x, batch=batch)
            Z_std = torch.exp(0.5 * Z_log_var)

            # C ∼ P(C|X) = N(Z_mean, Z_std)
            condition = Z_mean + Z_std * torch.randn_like(Z_mean)

            kld = -0.5 * torch.sum(1 + Z_log_var - Z_mean.pow(2) - Z_log_var.exp())
        else:
            condition = self.sdm.inverse(x, batch)

        # t ~ U(0, 1)
        t = torch.rand((batch_size, 1), device=x.device).to(x.device)

        # p_0t(x) = N(μ, σ) = q(\tilde{x} | x) 
        mean, std = self.pdm.marginal_prob_params(x, t, batch, condition)

        # \tilde{x} ~ p_0t(x) 
        eps = torch.randn_like(x)
        x_tilde = mean + std * eps

        # s(x + sigma * eps, t)
        score = self.pdm.score(x=x_tilde, t=t, batch=batch, condition=condition)

        # s(x + sigma * eps, t) - ∇ log p(\tilde{x} | x)
        # see slides for derivation
        loss = 0.5 * (score * std + eps).pow(2).sum(-1)

        # aggregate over batches
        loss = loss.mean()

        if self.lambda_kld > 0:
            loss += self.lambda_kld * kld

        return loss

    def latent_diffusion_loss(self, train_batch, batch_idx):
        """This method computes the loss for the latent diffusion phase (training of LTN) of the PointScoreDiffusion model.

        Args:
            train_batch (torch_geometric.data.Data): Training batch containing input data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value for the latent diffusion phase.

        """

        x, batch = train_batch.pos, train_batch.batch
        batch_size = batch.max() + 1

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            Z_mean, Z_log_var = self.sdm.get_params(x, batch=batch)
            Z_std = torch.exp(0.5 * Z_log_var)

            # C ∼ P(C|X) = N(Z_mean, Z_std)
            condition = Z_mean + Z_std * torch.randn_like(Z_mean)
        else:
            condition = self.sdm.inverse(x, batch)

        condition_batch = torch.arange(0, condition.shape[0])

        # t ~ U(0, 1)
        t = torch.rand((batch_size, 1), device=x.device).to(x.device)

        # p_0t(x) = N(μ, σ) = q(\tilde{x} | x)
        mean, std = self.ltn.marginal_prob_params(condition, t, condition_batch)

        # # \tilde{x} ~ p_0t(x) 
        eps = torch.randn_like(condition)
        condition_tilde = mean + std * eps

        # s(x + sigma * eps, t)
        score = self.ltn.score(x=condition_tilde, t=t, batch=condition_batch)

        # s(x + sigma * eps, t) - ∇ log p(\tilde{x} | x)
        # see slides for derivation
        loss = 0.5 * (score * std + eps).pow(2).sum(-1)

        # # aggregate over batches
        loss = loss.mean()

        return loss

    def _relaxed_distortion_measure(self, func, z, eta=0.2, augment_z=True):
        """Calculate the relaxed distortion measure for a given generative function and latent representation.

        Args:
            func (Callable): Generative function to evaluate relaxed distortion measure on.
            z (torch.Tensor): Latent representation tensor (condition).
            eta (float): Parameter controlling augmentation strength. Default is 0.2.
            augment_z (bool): Flag indicating whether to apply latent representation augmentation. Default is True.

        Returns:
            torch.Tensor: Relaxed distortion measure value.

        Source: https://github.com/Gabe-YHLee/IRVAE-public
        """

        bs = z.size(0)
        z_dim = z.size(1)

        # z ~ interp([z_1-b_1, z_2+b_2]), with z_1, z_2 ∈ batch 
        # if true sample z using linear (inter/extra)polation
        if augment_z:
            assert bs > 1, "can not use Z augmentation if bs < 2"
            z_permuted = z[torch.randperm(bs)]
            alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha * z + (1 - alpha) * z_permuted

        else:
            z_augmented = z

        # loss
        v = torch.randn(bs, z_dim).to(z_augmented)

        # Tr[G(z)] 
        Jv = torch.autograd.functional.jvp(func, z_augmented, v=v, create_graph=True)[1]
        TrG = torch.sum(Jv.view(bs, -1) ** 2, dim=1).mean()

        # Tr[G(z)^2]
        JTJv = (
            torch.autograd.functional.vjp(func, z_augmented, v=Jv, create_graph=True)[1]
        ).view(bs, -1)
        TrG2 = torch.sum(JTJv**2, dim=1).mean()

        return TrG2 / TrG**2

    def latent_metric_loss(
        self, train_batch, batch_idx, n_samples=100, lambda_reg=1e-1
    ):
        """Compute the total loss, reconstruction loss, and distortion loss for the latent metric learning.

        Args:
            train_batch (torch_geometric.data.Data): Input batch containing vertices and batch information.
            batch_idx (int): Batch index.
            n_samples (int): Number of samples for metric loss computation. Default is 100.
            lambda_reg (float): Regularization parameter for distortion_measure. Default is 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss, and distortion measure.
        """

        x, batch = train_batch.pos, train_batch.batch

        if self.stn is not None:
            x = self.stn(x, batch)

        if self.lambda_kld > 0:
            C_mean, C_log_var = self.sdm.get_params(x, batch=batch)
            C_std = torch.exp(0.5 * C_log_var)
            C = C_mean + C_std * torch.randn_like(C_mean)
        else:
            C = self.sdm.inverse(x, batch)

        Z_metric = self.mtn.inverse(C)

        # REC LOSS
        C_rec = self.mtn.forward(Z_metric)
        rec_loss = (C - C_rec).pow(2).mean(-1)
        rec_loss = rec_loss.mean()

        # METRIC LOSS
        z_template = 0.8 * torch.randn(
            n_samples, 3, generator=torch.Generator().manual_seed(42)
        ).repeat(batch.max() + 1, 1).to(x)

        batch = (
            torch.concat([i * torch.ones(n_samples) for i in range(batch.max() + 1)])
            .to(x)
            .long()
        )

        f_generator = lambda z: self.forward(
            Z=self.mtn.forward(z),
            apply_pdm=True,
            time_steps=6,
            z=z_template,
            batch=batch,
        )[1].view(z.shape[0], -1, 3)

        metric_loss = (
            lambda_reg
            * self._relaxed_distortion_measure(f_generator, Z_metric, augment_z=True).mean()
        )

        # TOTAL LOSS
        loss = rec_loss + metric_loss

        return loss, rec_loss, metric_loss


    def training_step(self, train_batch, batch_idx):
        """Training step for the PyTorch Lightning module based on the current training phase.

        Args:
            train_batch (DataBatch): Input batch for training.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value computed based on the current training phase.

        """

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

    # VALIDATION STEP
    def latent_metric_validation_loss(self, train_batch, batch_idx, n_cps=6, n_samples=8000):
        """Compute the geodesic energy and reconstruction error for the latent metric learning during validation.

        Args:
            train_batch (torch_geometric.data.Data): Input batch containing position and batch information.
            batch_idx (int): Batch index.
            n_cps (int): Number of control points for geodesic energy computation. Default is 6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Geodesic energy and reconstruction error.

        """
        x, batch = train_batch.pos, train_batch.batch

        assert batch.max() == 1, "TODO: Batch this function."

        if self.stn is not None:
            x = self.stn(x, batch)

        with torch.no_grad():
            # start and end of interpolation in latent space: [C*_0, C*_T]
            C = self.sdm.inverse(x, batch)

            # start and end of interpolation in metric space: [Z_0, Z_T]
            Z = self.mtn.inverse(C)

            # interpolation in metric space: [Z_0, Z_1, ..., Z_T]
            Z_interp = torch.lerp(
                input=Z[:1],
                end=Z[1:],
                weight=torch.linspace(0, 1, n_cps)[:, None].to(x.device),
            )

            # interpolationin latent space [C_0, C_1, ..., C_T]
            C_interp = self.mtn.forward(Z_interp)
            reconstruction_error = (C - C_interp[[0, -1]]).pow(2).mean(-1)

            # replace start and end point by known point 
            C_interp = torch.concat([C[:1], C_interp[1:-1], C[1:]])

            # setup template
            z_template = 0.8 * torch.randn(n_samples, 3).repeat(n_cps, 1).to(x)
            batch_template = (
                torch.concat([i * torch.ones(n_samples) for i in range(n_cps)]).to(x).long()
            )

            # generate interpolated shapes
            Xs = self.forward(
                Z=C_interp,
                apply_pdm=True,
                time_steps=10,
                n_samples=8000,  
                z=z_template,
                batch=batch_template,
            )[1].view(n_cps, -1, 3)

            # E(γ) = 0.5 * (∫g(γ˙(t), γ˙(t))) dt
            delta_t = 1 / (n_cps - 1)
            geodesic_energy = (
                0.5 * (Xs[:-1] - Xs[1:]).pow(2).sum(-1).mean(-1).div(delta_t).sum()
            )

        return geodesic_energy, reconstruction_error

    def validation_step(self, valid_batch, batch_idx):
        """Validation step for the PyTorch Lightning module based on the current training phase.

        Args:
            valid_batch (DataBatch): Input batch for validation.
            batch_idx (int): Batch index.

        Returns:
            None
        """

        if self.phase == Phase.TRAIN_POINT_DIFFUSION:
            x, batch = valid_batch.pos, valid_batch.batch

            with torch.no_grad():
                ll = self.log_likelihood(x, batch, time_steps=10, apply_stn=True)

            self.log("valid_point_ll", ll.mean(), batch_size=valid_batch.num_graphs)

        elif self.phase == Phase.TRAIN_LATENT_DIFFUSION:
            x, batch = valid_batch.pos, valid_batch.batch

            with torch.no_grad():
                ll_Z = self.log_prob_Z(x, batch, time_steps=10, apply_stn=True, estimate_trace=True)

            self.log("valid_latent_ll", ll_Z.mean(), batch_size=valid_batch.num_graphs)

        elif self.phase == Phase.TRAIN_METRIC_TRANSFORMER:
            ge, re = self.latent_metric_validation_loss(valid_batch, batch_idx)
            self.log(
                "valid_metric_geodesic", ge.mean(), batch_size=valid_batch.num_graphs
            )
            self.log(
                "valid_metric_reconstruction",
                re.mean(),
                batch_size=valid_batch.num_graphs,
            )
