#!/usr/bin/env python3

import torch
from gembed.core.module.stochastic import AbstractSDE
from gembed.core.distribution import MultivariateNormal


class VPSDE(AbstractSDE):
    """Variance preserving Stochastic differential equation, this is the continuous version of the DDPM model.

    Source: This code is based on https://github.com/yang-song/score_sde/blob/main/sde_lib.py
    """

    def __init__(self, beta_min, beta_max, dim, **kwargs):
        base_distribution = MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim, dim),
        )

        super().__init__(dim=dim, base_distribution=base_distribution, **kwargs)

        print(f"VPSDE(β_min: {beta_min}, β_max: {beta_max})")

        self.beta_min = beta_min
        self.beta_max = beta_max

        self.beta = lambda t: torch.lerp(
            torch.Tensor([self.beta_min]).to(t.device),
            torch.Tensor([self.beta_max]).to(t.device),
            t,
        )

    def drift_coefficient(self, x, t, batch=None, condition=None):
        """ Method to define the drift coefficient.
        
        \begin{equation}
            f(x, t) = -1/2 * βₜ * x
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Drift coefficient.
        """

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor([None, t])
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        # -1/2 * βₜ * x
        return (-0.5 * self.beta(t)).to(x.device)[batch] * x

    def diffusion_coefficient(self, x, t, batch=None, condition=None):
        """ Abstract method to define the diffusion coefficient.

        \begin{equation}
            G(x, t) = √βₜ
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Diffusion coefficient.
        """

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor(t)
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        result = torch.sqrt(self.beta(t)).to(t.device)
        return result[batch]

    def marginal_prob_params(self, x, t, batch=None, condition=None):
        """ Method to define the parameters for the marginal probability distribution.
    
        returns μ, σ for p_0t(x) = N(μ, σ)

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tuple: Tuple containing mean and standard deviation parameters.
        """

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor(t)
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )[batch]
        mean = x * torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))

        return mean, std
