#!/usr/bin/env python3

from abc import abstractmethod
from typing import List, Tuple, Union

import torch
from torch import Tensor
from gembed.core.module.bijection import ODE
from gembed.core.module import InvertibleModule
from gembed.core.module.bijection import ContinuousAmbientFlow
from gembed.core.distribution import MultivariateNormal
from gembed.core.module.bijection.continuous_ambient_flow import CAFDynamics
import lightning as pl


class DensityDynamics(CAFDynamics):
    def __init__(
        self,
        score_fn,
        diffusion_coefficient,
        inverse_drift_coefficient,
        estimate_trace: bool = False,
    ):

        super().__init__(fdyn=score_fn)
        self.score_fn = score_fn
        self.diffusion_coefficient = diffusion_coefficient
        self.inverse_drift_coefficient = inverse_drift_coefficient

        self.estimate_trace = estimate_trace

    def forward(
        self,
        t: Tensor,
        states: Union[List[Tensor], Tensor],
        c: Union[List[Tensor], Tensor, None] = None,
        noise: Union[Tensor, None] = None,
        batch: Union[Tensor, None] = None,
        **kwargs,
    ) -> List[Tensor]:

        pos, *_ = states

        # estimate trace => noise is not None
        assert not self.estimate_trace or noise is not None, (
            f"Noise should be passed to dynamic forward when ",
            f"using hutchinson trace estimator.",
        )

        with torch.set_grad_enabled(True):
            # t shape([nbatches, 1])
            t = t[None].repeat(batch.max() + 1, 1)
            x = pos.requires_grad_(True)

            drift = self.inverse_drift_coefficient(x=x, t=t, batch=batch, condition=c)
            div_drift, _ = self.evaluate_trace(drift, x, noise=noise)

        return drift, div_drift


class AbstractSDE(pl.LightningModule, InvertibleModule):
    """ Abstract Stochastic Differential Equation (SDE) class for defining continuous generative models using normalizing flows.
        
        This code is based on source: https://github.com/yang-song/score_sde/blob/main/sde_lib.py and correspondeing paper
    """
    #TODO: split this in a SDE and normalising flow component
    def __init__(self, f_score, dim, base_distribution, sample_method="ode", sample_method_kwargs={}):
        super().__init__()

        self.dim = dim
        self.f_score = f_score
        self.base_distribution = base_distribution

        self.set_sampler(sample_method, **sample_method_kwargs)

        self.density_estimator = ContinuousAmbientFlow(
            DensityDynamics(
                self.score, self.diffusion_coefficient, self.inverse_drift_coefficient
            ),
            noise_distribution=MultivariateNormal(torch.zeros(dim), torch.eye(dim, dim)),
            estimate_trace=False,
            method="rk4",
        )

    def set_sampler(self, sample_method, **sample_method_kwargs):
        """ Set the sampler based on the chosen sampling method.

        Args:
            sample_method (str): Sampling method for the SDE.
            sample_method_kwargs (dict): Additional keyword arguments for the chosen sampling method.
        """
        if sample_method == "ode":
            self.sampler = ODE(
                lambda t, x, c, batch: self.inverse_drift_coefficient(
                    x=x, t=t.repeat(batch.max() + 1, 1), batch=batch, condition=c
                ),
                **sample_method_kwargs,
                method="rk4",
            )

    def inverse_drift_coefficient(self, x, t, batch=None, condition=None):
        """
            Calculate the inverse drift coefficient.

            \begin{equation}
            f(x, t) - \frac{1}{2} g(t)^2 \nabla _x \log p_t(x)
            \end{equation}

            Args:
                x (Tensor): Input tensor.
                t (Tensor): Time tensor.
                batch (Tensor): Batch tensor (default is None).
                condition (Tensor): Condition tensor (default is None).

            Returns:
                Tensor: Inverse drift coefficient.
        """
        drift = self.drift_coefficient(x, t, batch, condition)
        diffusion = self.diffusion_coefficient(x, t, batch, condition)

        score = self.score(x, t, batch, condition)

        # f(x, t) - 1/2 g(t)^2 ∇ₓlog pₜ(x)
        return drift - (0.5 * diffusion ** 2 * score)

    def inverse_diffusion_coefficient(self, x, t, batch=None, condition=None):
        """ Calculate the inverse diffusion coefficient.

        \begin{equation}
            g(x, t)
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Inverse diffusion coefficient.

        """
        
        # g(t)
        diffusion = self.diffusion_coefficient(x, t, batch, condition)

        return diffusion

    def log_prob(self, x, batch, condition, time_steps=100, estimate_trace=False):
        """ Calculate the log probability of the data.

        Args:
            x (Tensor): Input tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).
            time_steps (int): Number of time steps for estimating the density (default is 100).
            estimate_trace (bool): Whether to estimate the trace (default is False).

        Returns:
            Tensor: Log probability of the data.
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        previous_estimate_trace = self.density_estimator.estimate_trace
        self.density_estimator.set_estimate_trace(estimate_trace)

        z, d_log_pz = self.density_estimator.forward(
            z=x, batch=batch, condition=condition, time_steps=time_steps
        )

        self.density_estimator.set_estimate_trace(previous_estimate_trace)

        # log px = log pz + ∫_t0^t1 -Tr[Jf] dt
        return self.base_log_prob(z) + d_log_pz

    @abstractmethod
    def drift_coefficient(self, x, t, batch=None, condition=None):
        """ Abstract method to define the drift coefficient.
        
        \begin{equation}
            f(x, t)
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Drift coefficient.
        """

        # R^d -> R^d
        raise NotImplemented()

    @abstractmethod
    def diffusion_coefficient(self, x, t, batch=None, condition=None):
        """ Abstract method to define the diffusion coefficient.

        \begin{equation}
            G(x, t)
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Diffusion coefficient.
        """

        # R -> R
        raise NotImplemented()

    @abstractmethod
    def base_log_prob(self, z):
        """ Method to compute the log density of the base distribution.

        \begin{equation}
            p(z) = N(0, I)
        \end{equation}

        Args:
            z (Tensor): Input tensor.

        Returns:
            Tensor: Log probability of the base distribution.
        """
        return self.base_distribution.log_prob(z)

    def sample_base(self, n_samples):
        """ Abstract method to sample from the base distribution.

        \begin{equation}
            z \sim p(z)
        \end{equation}

        Args:
            shape: Shape of the sample.

        Returns:
            Tensor: Sample from the base distribution.
        """
        return self.base_distribution.sample(n_samples)

    @abstractmethod
    def marginal_prob_params(self, x, t, batch=None, condition=None):
        """ Abstract method to define the parameters for the marginal probability distribution.
    
        returns μ, σ for p_0t(x) = N(μ, σ)

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tuple: Tuple containing mean and standard deviation parameters.
        """

        raise NotImplemented()

    def score(self, x, t, batch=None, condition=None):
        """ Calculate the score for the SDE.

        \begin{equation}
            s(x, t, c)
        \end{equation}

        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).

        Returns:
            Tensor: Score for the SDE.
        """
        return self.f_score(x=x, t=t, c=condition, batch=batch)

    def forward(
        self,
        z: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        time_steps: int = 100,
        return_time_steps: bool = False,
        **kwargs,
    ):
        """ Forward pass through the model to generate samples.

        Args:
            z (Tensor): Input tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).
            time_steps (int): Number of time steps for sampling (default is 100).
            return_time_steps (bool): Whether to return intermediate time steps (default is False).

        Returns:
            Tensor: Generated samples.
        """

        if batch is None:
            batch = torch.zeros(z.shape[0], dtype=torch.long).to(z.device)

        # ∫f(x, t) - g(t)^2 ∇log pt(x) dt + g(t)dw
        return self.sampler.inverse(
            z, batch, condition, time_steps, return_time_steps, **kwargs
        )

    def inverse(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        time_steps: int = 100,
        return_time_steps: bool = False,
        **kwargs,
    ):
        """ Inverse pass through the model to generate representation.

        Args:
            x (Tensor): Input tensor.
            batch (Tensor): Batch tensor (default is None).
            condition (Tensor): Condition tensor (default is None).
            time_steps (int): Number of time steps for inverse sampling (default is 100).
            return_time_steps (bool): Whether to return intermediate time steps (default is False).

        Returns:
            Tensor: Reconstructed samples.
        """
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        return self.sampler.forward(
            x, batch, condition, time_steps, return_time_steps, **kwargs
        )

    # def forward(
    #     self,
    #     z: Tensor,
    #     batch: Union[Tensor, None] = None,
    #     condition: Union[Tensor, None] = None,
    #     time_steps: int = 8,
    #     return_time_steps: bool = False,
    #     **kwargs,
    # ):
    #     """Integrate dynamics forward in time from $t \in [0, 1]$. This results in the ODE

    #     \begin{equation}
    #       x = z + \int_{0}^{1} g_\phi(t, z_t) \; dt.
    #     \end{equation}

    #     Where $g_\phi(t, z_t)$ the dynamics.

    #     """

    #     x = z
    #     dt = torch.Tensor([1 / (time_steps - 1)]).to(x.device)

    #     objects = []
    #     for t in torch.linspace(1, 0, time_steps):
    #         t = t[None].to(x.device)
    #         f = self.drift_coefficient(x=z, t=t) * dt
    #         G = self.diffusion_coefficient(x, t) * torch.sqrt(dt)

    #         rev_f = f - G ** 2 * self.score(x, t)
    #         rev_G = G

    #         x_mean = x - rev_f

    #         z = torch.randn_like(x)
    #         x = x_mean + rev_G * z

    #         objects.append(x)
    #         #objects.append(x_mean)

    #     if return_time_steps:
    #         return objects

    #     return objects[-1]

    # def inverse(
    #     self,
    #     x: Tensor,
    #     batch: Union[Tensor, None] = None,
    #     condition: Union[Tensor, None] = None,
    #     time_steps: int = 8,
    #     return_time_steps: bool = False,
    #     **kwargs,
    # ):
    #     dt = torch.Tensor([1 / (time_steps - 1)])

    #     objects = []
    #     for t in torch.linspace(0, 1, time_steps):
    #         z = torch.randn(x.shape[0], 3)
    #         f = self.drift_coefficient(x=x, t=t)
    #         G = self.diffusion_coefficient(x, t)

    #         # dx = f(x, t) dt + g(t) dw
    #         x = x + (f*dt) + (G * z * torch.sqrt(dt))

    #         objects.append(x)

    #     if return_time_steps:
    #         return objects

    #     return objects[-1]
