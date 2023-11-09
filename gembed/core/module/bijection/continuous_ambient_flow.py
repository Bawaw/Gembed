#!/usr/bin/env python3

from typing import List, Tuple, Union

import torch
from gembed.core.distribution import DistributionProtocol
from gembed.core.module.bijection.abstract_ode import AbstractODE
from gembed.numerics.trace import hutchinson_trace_estimator, trace
from torch import Tensor, nn


class CAFDynamics(nn.Module):
    def __init__(self, fdyn: nn.Module, estimate_trace: bool = False):

        # TODO: make fdyn callable
        super().__init__()
        self.fdyn = fdyn
        self.estimate_trace = estimate_trace

    def evaluate_trace(
        self, x_out: Tensor, x_in: Tensor, noise: Union[Tensor, None]
    ) -> Tuple[Tensor, Tensor]:

        if self.estimate_trace:
            trJ, e_dzdx = hutchinson_trace_estimator(x_out, x_in, noise)
        else:
            trJ, e_dzdx = trace(x_out, x_in)

        return trJ, e_dzdx

    def forward(
        self,
        t: Tensor,
        states: Union[List[Tensor], Tensor],
        c: Union[List[Tensor], Tensor, None] = None,
        noise: Union[Tensor, None] = None,
        **kwargs,
    ) -> List[Tensor]:

        """Return the tuple (g_\phi(x_t), -Tr \left \{ J_{g_\phi}\right \}(x_t)).

        If estimate_density is true, the density is estimated using Hutchinson trace estimation which states that
        \begin{equation}
          Tr\{A\} = \mathbb{E}_{z\sim p(z)} [ \epsilon^T A \epsilon ]
        \end{equation},
        as long as $p(\epsilon)$ has a zero mean and unit variance. The value for $\epsilon$ is fixed for each solve.
        """

        pos, *_ = states

        # estimate trace => noise is not None
        assert not self.estimate_trace or noise is not None, (
            f"Noise should be passed to dynamic forward when ",
            f"using hutchinson trace estimator.",
        )

        with torch.set_grad_enabled(True):
            pos = pos.requires_grad_(True)

            pos_dot = self.fdyn.forward(t, pos, c, **kwargs)

            # compute/approximate trace
            trJ, _ = self.evaluate_trace(pos_dot, pos, noise=noise)

        return pos_dot, -trJ


class ContinuousAmbientFlow(AbstractODE):
    def __init__(
        self,
        dynamics: Union[CAFDynamics, nn.Module],
        estimate_trace: bool = False,
        noise_distribution: Union[DistributionProtocol, None] = None,
        **kwargs,
    ):

        if not isinstance(dynamics, CAFDynamics):
            assert isinstance(
                dynamics, nn.Module
            ), f"Expected, dynamics to be of type Dynamics or torch.nn.Module"
            dynamics = CAFDynamics(dynamics)

        super().__init__(dynamics, **kwargs)

        # noise distribution used for trace estimator
        self.noise_distribution = noise_distribution
        self.set_estimate_trace(estimate_trace)

    @property
    def estimate_trace(self) -> bool:
        return self.dynamics.estimate_trace

    def set_estimate_trace(self, estimate_trace: bool):
        assert not (estimate_trace and self.noise_distribution is None), (
            f"If trace is estimated a noise distribution should be set for ",
            f"{self.__class__.__name__}",
        )

        self.dynamics.estimate_trace = estimate_trace

    def forward(self, **kwargs):
        """Integrate dynamics in the forward in time from $t \in [0, 1]$. """

        # z=f(x), -log |det Jf|
        x, log_det_jac_f = super().forward(**kwargs)

        # compute the change in log density
        # (Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling and Inference." J. Mach. Learn. Res. 22.57 (2021): 1-64.)
        # log px = log pz - log |det Jf|
        #        => log pz + (-log |det Jf|)
        d_log_p = log_det_jac_f

        # concat output
        return (x, d_log_p)

    def inverse(self, **kwargs):

        """Integrate dynamics in the backward in time from $t \in [1, 0]$. """

        z, log_det_jac_f_inv = super().inverse(**kwargs)

        # compute the change in log density
        # (Papamakarios, George, et al. "Normalizing Flows for Probabilistic Modeling and Inference." J. Mach. Learn. Res. 22.57 (2021): 1-64.)
        # log px = log pz + log |det Jf_inv|
        #        => log pz - (-log |det Jf_inv|)
        #        => log pz + div(Jf_inv)
        d_log_p = -log_det_jac_f_inv

        # concat output
        return (z, d_log_p)

    def integrate(
        self,
        pos: Tensor,
        t_span: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        return_time_steps: bool = False,
        **kwargs,
    ):

        """Integrate the dynamics and compute the change in log density, kinetic energy, and Frobenius norm of the Jacobian by solving the following ODE:

            \begin{bmatrix}
            x_{t_1}\\
            \Delta p\\
            \Delta e\\
            \Delta n
            \end{bmatrix}
            =
            \begin{bmatrix}
            x_{t_0}\\
            0 \\
            0 \\
            0
            \end{bmatrix}
            +
            \int_{t_0}^{t_1}
            \begin{bmatrix}
            g_\phi(t, x_t)\\
            -Tr \left[ J_{g_\phi} (x_t) \right] \\
            - \lVert g_\phi(x_t) \rVert^2 \\
            - \lVert J_{g_\phi}(x_t) \rVert_F^2
            \end{bmatrix}
            \; dt

        Args:
            pos (Tensor): The initial position tensor.
            t_span (Tensor): The time span to integrate over.
            batch (Union[Tensor, None], optional): The batch index assignment for the pos argument. Default: None.
            condition (Union[Tensor, None], optional): The conditional arguments used to supplement the dynamics function. Default: None.
            return_time_steps (bool, optional): Whether to return the full sequence of time steps or just the final one. Default: False.
            **kwargs: Additional arguments to pass to the odeint function.

        Returns:
            If return_time_steps is False, returns a tuple containing the final position, divergence. Otherwise, returns a tuple containing the full sequence of position, divergence over the time steps."""

        divergence = torch.zeros(pos.shape[0]).to(pos.device)

        # if estimating trace with hutchinson sample an epsilon
        epsilon = None
        if self.estimate_trace and self.noise_distribution is not None:
            epsilon = self.noise_distribution.sample(pos.shape[0])

        if condition is not None:
            if batch is None:
                batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

            assert condition.shape[0] == batch.max() + 1, (
                f"Mismatch between batch {batch.max()+1} ",
                f"and number of conditions {condition.shape[0]}.",
            )

            #condition = condition[batch]

        dynamics = lambda t, x: self.dynamics.forward(
            t, x, condition, noise=epsilon, batch=batch
        )

        pos, divergence = self.odeint(
            dynamics, (pos, divergence), t_span, self.dynamics.parameters()
        )

        if not return_time_steps:
            return pos[-1], divergence[-1]

        return pos, divergence
