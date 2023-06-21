#!/usr/bin/env python3

import torch
import numpy as np

def gradient_ascent(
    init_x, f_grad, condition=None, batch=None, n_steps=10, step_size=1e-4, batch_size=20000
):
    """
    Performs gradient ascent to maximize a function with respect to the input.

    Args:
        init_x (torch.Tensor): The initial input tensor.
        f_grad (callable): A function that computes the gradient of the objective function with respect to the input.
        condition (torch.Tensor): The conditional arguments used to supliment the gradient. Default: None.
        batch (torch.Tensor, optional): The batch index assignment for the init_x argument. Default: None.
        n_steps (int, optional): The number of gradient ascent steps to perform. Default: 10.
        step_size (float, optional): The step size to use in each gradient ascent step. Default: 1e-4.
        batch_size (int, optional): The size of each batch to use in computing the gradient of the objective function. Default: 20000.

    Returns:
        torch.Tensor: The final input tensor after performing gradient ascent.
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

            grad = f_grad(x, b, condition).detach()

            xs[i] = x + step_size * grad

    return torch.concat(xs)
