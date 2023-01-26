#!/usr/bin/env python3

from abc import abstractmethod
from typing import Union

import torch
from gembed.core.module import InvertibleModule
from torch import Tensor, nn
from torchdiffeq import odeint, odeint_adjoint


class AbstractODE(InvertibleModule):
    def __init__(
        self,
        dynamics: nn.Module,
        adjoint: bool = False,
        rtol=1e-7,
        atol=1e-9,
        method: str = "dopri5",
        options=None,
        event_fn=None,
    ):

        super().__init__()

        if adjoint:
            # if working with adjoint we need to let the ode know the parameters
            # of the ODE
            self.odeint = lambda fdyn, x, t_span, params: odeint_adjoint(
                fdyn,
                x,
                t_span,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
                event_fn=event_fn,
                adjoint_params=params,
            )

        else:
            self.odeint = lambda fdyn, x, t_span, _: odeint(
                fdyn,
                x,
                t_span,
                rtol=rtol,
                atol=atol,
                method=method,
                options=options,
                event_fn=event_fn,
            )

        self.dynamics = dynamics

    def forward(
        self,
        z: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        time_steps: int = 10,
        return_time_steps: bool = False,
        **kwargs
    ):
        """Integrate dynamics forward in time from $t \in [0, 1]$. This results in the ODE

        \begin{equation}
          x = z + \int_{0}^{1} g_\phi(t, z_t) \; dt.
        \end{equation}

        Where $g_\phi(t, z_t)$ the dynamics.

        """

        # t = [0, 1]
        t_span = torch.linspace(0, 1, time_steps).to(z.device)

        output = self.integrate(
            pos=z,
            t_span=t_span,
            batch=batch,
            condition=condition,
            return_time_steps=return_time_steps,
            **kwargs
        )

        return output

    def inverse(
        self,
        x: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        time_steps: int = 10,
        return_time_steps: bool = False,
        **kwargs
    ):

        """Integrate dynamics in the backward in time from $t \in [1, 0]$. This results in the ODE

        \begin{equation}
          z = x + \int_{1}^{0} g_\phi(t, x_t) \; dt.
        \end{equation}

        Where $g_\phi(t, x_t)$ the dynamics.

        """

        t_span = torch.linspace(1, 0, time_steps).to(x.device)
        output = self.integrate(
            pos=x,
            t_span=t_span,
            batch=batch,
            condition=condition,
            return_time_steps=return_time_steps,
            **kwargs
        )

        return output

    @abstractmethod
    def integrate(
        self,
        pos: Tensor,
        t_span: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        return_time_steps: bool = False,
        **kwargs
    ):

        """ Integrate dynamics. """

        return NotImplemented
