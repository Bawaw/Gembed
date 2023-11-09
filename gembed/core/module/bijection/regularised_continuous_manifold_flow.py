#!/usr/bin/env python3

from typing import List, Tuple, Union
from gembed.core.module import InvertibleModule

import torch
from gembed.core.distribution import DistributionProtocol
from gembed.core.module.bijection.abstract_ode import AbstractODE
from gembed.numerics.trace import hutchinson_trace_estimator, trace
from torch import Tensor, nn


# TODO: documentation
class RCMFDynamics(nn.Module):
    def __init__(self, fdyn: nn.Module):
        super().__init__()
        self.fdyn = fdyn

    def forward(
        self,
        t: Tensor,
        states: Union[List[Tensor], Tensor],
        c: Union[List[Tensor], Tensor, None] = None,
        **kwargs,
    ) -> List[Tensor]:

        """Return the tuple  \big(g_\phi(x_t), -Tr \left [ J_{g_\phi}(x_t)\right], \lVert g_\phi(x_t) \rVert^2, \lVert J_{g_\phi}(x_t) \rVert_F^2 \big)

        If estimate_density is true, the density is estimated using Hutchinson trace estimation;
        \begin{equation}
          Tr\{A\} = \mathbb{E}_{z\sim p(z)} [ \epsilon^T A \epsilon ]
        \end{equation},
        as long as $p(\epsilon)$ has a zero mean and unit variance. The value for $\epsilon$ is fixed for each solve.

        """

        pos, *_ = states

        with torch.set_grad_enabled(True):
            pos = pos.requires_grad_(True)

            pos_dot = self.fdyn.forward(t, pos, c, **kwargs)

            # kinetic energy
            e_dot = torch.sum(pos_dot ** 2, 1)

        return pos_dot, -e_dot


class ZeroProject(InvertibleModule):
    def forward(self, z):
        # f: x -> [x | 0]
        return torch.cat([z, torch.zeros(z.shape[0], 1).to(z.device)], -1)

    def inverse(self, x):
        # f-1: [x | v] -> x
        return x[:, :-1]

    def __str__(self):
        return str(self.__class__.str())


class RegularisedContinuousManifoldFlow(AbstractODE):
    def __init__(
        self,
        dynamics: Union[RCMFDynamics, nn.Module],
        chart=None,
        compute_log_density: bool = False,
        **kwargs,
    ):
        if not isinstance(dynamics, RCMFDynamics):
            assert isinstance(
                dynamics, nn.Module
            ), f"Expected, dynamics to be of type Dynamics or torch.nn.Module"
            dynamics = RCMFDynamics(dynamics)

        super().__init__(dynamics, **kwargs)

        if chart is None:
            self.chart = ZeroProject()
        else:
            self.chart = chart

        self.set_compute_log_density(compute_log_density)

    @property
    def compute_log_density(self) -> bool:
        return self._compute_log_density

    def set_compute_log_density(self, compute_log_density: bool):
        self._compute_log_density = compute_log_density

    def _forward(self, z, **kwargs):
        x0 = self.chart.forward(z=z)
        x1, *combined_dynamics = super().forward(x0, **kwargs)

        return x1, *combined_dynamics

    def forward(self, z, include_combined_dynamics=False, **kwargs):
        """Integrate dynamics in the forward in time from $t \in [0, 1]$. """

        # z=f(x), -log |det Jf|
        x, *combined_dynamics = self._forward(z=z, **kwargs)

        if self.compute_log_density:
            raise NotImplementedError()
        else:
            d_log_p = torch.zeros(x.shape[0]).to(x.device)

        output = (x, d_log_p)

        if include_combined_dynamics:
            output += tuple(combined_dynamics)

        return output

    def _inverse(self, x, **kwargs):
        z0, *combined_dynamics = super().inverse(x=x, **kwargs)
        z1 = self.chart.inverse(z0)

        # PLOT #
        # x_rec, *_ = self._forward(z1, **kwargs)
        # from gembed.vis import plot_objects
        # plot_objects(
        #     (x.cpu(), None),
        #     (z0.cpu(), None),
        #     (z1.cpu(), None),
        #     (x_rec.cpu(), None),
        # )
        ########

        return z1, *combined_dynamics

    def inverse(self, x, include_combined_dynamics=False, **kwargs):

        """Integrate dynamics in the backward in time from $t \in [1, 0]$. """

        z, *combined_dynamics = self._inverse(x=x, **kwargs)

        if self.compute_log_density:
            # TODO:
            raise NotImplementedError()
        else:
            d_log_p = torch.zeros(x.shape[0]).to(x.device)

        output = (z, d_log_p)

        if include_combined_dynamics:
            output += tuple(combined_dynamics)

        return output

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
            \Delta \log p(x)\\
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
            If return_time_steps is False, returns a tuple containing the final position, divergence, kinetic energy, and norm of the Jacobian. Otherwise, returns a tuple containing the full sequence of position, divergence, kinetic energy, and norm of the Jacobian over the time steps."""

        kinetic_energy = torch.zeros(pos.shape[0]).to(pos.device)

        # if estimating trace with hutchinson sample an epsilon
        if condition is not None:
            if batch is None:
                batch = torch.zeros(pos.shape[0], dtype=torch.long).to(pos.device)

            assert condition.shape[0] == batch.max() + 1, (
                f"Mismatch between batch {batch.max()+1} ",
                f"and number of conditions {condition.shape[0]}.",
            )

            #condition = condition[batch]

        dynamics = lambda t, x: self.dynamics.forward(t, x, condition, batch=batch)

        pos, kinetic_energy = self.odeint(
            dynamics,
            (pos, kinetic_energy),
            t_span,
            self.dynamics.parameters(),
        )

        if not return_time_steps:
            return pos[-1], kinetic_energy[-1]

        return pos, kinetic_energy
