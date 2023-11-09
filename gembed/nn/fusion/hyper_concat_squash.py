#!/usr/bin/env python3

import math
import torch
from torch import nn


class HyperConcatSquash(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        context_dim,
        t_dim,
        hidden_dim,
        hyper_bias=None,
        hyper_gate=None,
        hyper_net=None,
    ):
        super().__init__()

        self.in_dim, self.out_dim, self.context_dim, self.t_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            context_dim,
            t_dim,
            hidden_dim,
        )
        self.W_shape = (self.out_dim, self.in_dim)

        if hyper_bias is None:
            # self.hyper_bias = nn.Sequential(
            #     nn.Linear(self.t_dim, self.out_dim),
            # )
            self.hyper_bias = nn.Sequential(
                # nn.LayerNorm(self.t_dim),
                nn.Linear(self.t_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_bias = hyper_bias

        if hyper_gate is None:
            # self.hyper_gate = nn.Sequential(
            #     nn.Linear(self.t_dim, self.out_dim),
            # )
            self.hyper_gate = nn.Sequential(
                # nn.LayerNorm(self.t_dim),
                nn.Linear(self.t_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_gate = hyper_gate

        if hyper_net is None:
            # self.hyper_net = nn.Sequential(
            #     nn.Linear(
            #         self.context_dim,
            #         math.prod(self.W_shape),
            #     ),
            # )
            self.hyper_net = nn.Sequential(
                # nn.Linear(2 * 3, 128),
                # nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Softplus(),
                # nn.ELU(),
                nn.Linear(
                    self.hidden_dim,
                    math.prod(self.W_shape),
                ),
            )
        else:
            self.hyper_net = hyper_net

    # def forward(self, x, c, t, batch):

    #     # get weight matrices
    #     W = self.hyper_net(c).view(self.W_shape)
    #     a = torch.sigmoid(self.hyper_gate(t.view(1, self.t_dim)))
    #     b = self.hyper_bias(t.view(1, self.t_dim))

    #     # evaluate layer: (a * (x @ W.T)) + b
    #     return (a * (x @ W.T)) + b

    def forward(self, x, c, t, batch):
        n_batch = batch.max() + 1

        assert c.shape[0] == n_batch, """Batch and context do not match."""

        # get weight matrices
        W = self.hyper_net(c).view(n_batch, *self.W_shape)
        a = torch.sigmoid(self.hyper_gate(t.view(n_batch, self.t_dim)))
        b = self.hyper_bias(t.view(n_batch, self.t_dim))

        # evaluate layer: (a * (x @ W.T)) + b
        batch_sizes = list(batch.unique_consecutive(return_counts=True)[1])
        result = torch.concat(
            [a[i] * (_x @ W[i].T) + b[i] for i, _x in enumerate(x.split(batch_sizes))]
        )  # TODO: can we make this more efficient using broadcasting?

        return result
