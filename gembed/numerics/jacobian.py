#!/usr/bin/env python3

import numpy as np
import torch

def batch_jacobian(y, x):
    """ Compute jacobian in batch format

    input:
        y: torch.Size([B, d2])
        x: torch.Size([B, d1])

    output:
        jacobian dy/dx: torch.Size([B, d2, d1])
    """

    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [torch.autograd.grad(y[:, i], x,
                               grad_outputs=vector,
                               retain_graph=True,
                               create_graph=True)[0].view(batch, -1)
                for i in range(single_y_size)]

    jac = torch.stack(jac, dim=1)

    return jac
