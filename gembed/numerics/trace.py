#!/usr/bin/env python3

from typing import Tuple

import torch
from gembed.numerics.jacobian import batch_jacobian
from torch import Tensor
from torch.autograd import grad


def trace(x_out, x_in, return_jacobian=False) -> Tuple[Tensor, Tensor]:
    """Exact jacobian trace computed using autograd, optionally return Jacobian as vector if return_jacobian is True.

    \begin{equation}
    \left(Tr \left \{ J_f \right \}(x_t), J_f\right
    \end{equation}

    """

    if return_jacobian:
        # note that this is much more computationally expensive since
        # we need to evaluate the entire jacobian
        J = batch_jacobian(x_out, x_in)

        TrJ = torch.stack([j.trace() for j in J])

        # shift view to vector format
        J = J.view(J.shape[0], -1)

    else:
        # https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/models/cnf.py
        TrJ = 0.
        for i in range(x_in.shape[1]):
            TrJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
        J = None

    return TrJ, J

def hutchinson_trace_estimator(x_out, x_in, epsilon) -> Tuple[Tensor, Tensor]:
    """Returns Hutchinson's estimated Jacobian trace, aditionally return the vector Jacobian product.

    \begin{equation}
    \left(\epsilon^T Tr \left \{ J_f \right \}(x_t) \epsilon, \epsilon^T J_f \right)
    \end{equation}

    """

    # https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/models/cnf.py

    # vector jacobian product
    eT_J = torch.autograd.grad(x_out, x_in, epsilon, create_graph=True)[0]

    # trace of jacobian  e^T J e
    TrJ = torch.einsum('bi,bi->b', eT_J, epsilon)

    return TrJ, eT_J
