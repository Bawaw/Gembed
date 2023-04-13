#!/usr/bin/env python3

import torch
import numpy as np


def test_grad(x, model, batch, c):
    log_px = model.pdm.log_prob(x, batch, c)
    print(log_px.mean())

    return torch.autograd.grad(log_px.sum(), x, create_graph=True)[0]


def gradient_ascent(
    init_x, f_grad, c, model, batch=None, n_steps=10, step_size=1e-4, batch_size=20000
):

    xs = list(torch.split(init_x, batch_size))
    for _ in range(n_steps):
        for i in range(len(xs)):
            x = xs[i]

            # grad = f_grad(x, batch).detach()
            grad = test_grad(x, model, batch, c).detach()

            xs[i] = x + step_size * grad

    return torch.concat(xs)
