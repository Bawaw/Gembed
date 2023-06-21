#!/usr/bin/env python3

import math
import torch
import numpy as np

def gradient_langevin_dynamics(
    init_x, f_grad, condition=None, batch=None, n_steps=10, step_size=1e-4, batch_size=20000
):
    """
    Performs Carlo sampling to generate samples from a target distribution. This can be viewed as an optimisation algorithm with a stochastic component to it.

    Args:
        init_x (torch.Tensor): The initial input tensor.
        f_grad (callable): A function that computes the gradient of the objective function with respect to the input.
        condition (torch.Tensor, optional): The conditional arguments used to supplement the gradient. Default: None.
        batch (torch.Tensor, optional): The batch index assignment for the init_x argument. Default: None.
        n_steps (int, optional): The number of Langevin Monte Carlo steps to perform. Default: 10.
        step_size (float, optional): The step size to use in each Langevin Monte Carlo step. Default: 1e-4.
        batch_size (int, optional): The size of each batch to use in computing the gradient of the objective function. Default: 20000.

    Returns:
        torch.Tensor: The final input tensor after performing Langevin Monte Carlo sampling.
    """

    xs = list(torch.split(init_x, batch_size))

    if batch is not None:
        bs = list(torch.split(batch, batch_size))

    for _ in range(n_steps):
        for i in range(len(xs)):
            x = xs[i]

            if batch is not None:
                b = bs[i]
            else:
                b = None

            grad = f_grad(x, batch, condition).detach()

            xs[i] = (
                x + step_size * grad + math.sqrt(2 * step_size) * torch.randn_like(x)
            )

    return torch.concat(xs)
