#!/usr/bin/env python3

from typing import List, Tuple, Union

import torch
from gembed.core.distribution import DistributionProtocol
from gembed.core.module.bijection.abstract_ode import AbstractODE
from gembed.numerics.trace import hutchinson_trace_estimator, trace
from torch import Tensor, nn


class CAFDynamics(nn.Module):
    def __init__(self, fdyn: nn.Module, estimate_trace: bool = False):

        super().__init__()
        self.fdyn = fdyn
        self.estimate_trace = estimate_trace

    def evaluate_trace(
        self, x_out: Tensor, x_in: Tensor, noise: Union[Tensor, None]
    ) -> Tuple[Tensor, Tensor]:

        if not self.estimate_trace:
            trJ, e_dzdx = trace(x_out, x_in)
        else:
            trJ, e_dzdx = hutchinson_trace_estimator(x_out, x_in, noise)

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
        assert estimate_trace and self.noise_distribution is not None, (
            f"If trace is estimated a noise distribution should be set for ",
            f"{self.__class__.__name__}",
        )

        self.dynamics.estimate_trace = estimate_trace

    def integrate(
        self,
        pos: Tensor,
        t_span: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        return_time_steps: bool = False,
        **kwargs,
    ):

        """ Integrate dynamics and co-integrate the change in log density. This results in the ODE
        \begin{equation}
          \begin{bmatrix}
          x_{t_1}\\
          \log p(x)
          \end{bmatrix}
          =
          \begin{bmatrix}
          x_{t_0}\\
          \log p(x)
          \end{bmatrix}
          +
          \int_{t_0}^{t^1}
          \begin{bmatrix}
          g_\phi(t, x_t)\\
          -Tr \left\{ J_{g_\phi} \right\} (x_t)
          \end{bmatrix}
          \; dt
        \end{equation}

        Where $g_\phi(t, z_t)$ the dynamics, and
        \begin{equation}
          - \log \lvert \det J_f \rvert = \int_{t_0} ^{t_1} -Tr \left \{ J_{g_\phi}\right \}(x_t).
        \end{equation}

        This is also known as the negative divergence of $g_\phi$, denoted as $-div(g_\phi)$.

        """

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

            condition = condition[batch]

        dynamics = lambda t, x: self.dynamics.forward(
            t, x, condition, noise=epsilon, batch=batch
        )

        pos, divergence = self.odeint(
            dynamics, (pos, divergence), t_span, self.dynamics.parameters()
        )

        if not return_time_steps:
            return pos[-1], divergence[-1]

        return pos, divergence
