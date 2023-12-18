#!/usr/bin/env python3

from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from gembed.core.distribution import DistributionProtocol
from gembed.core.module.bijection.ode import ODE
from gembed.numerics.trace import hutchinson_trace_estimator, trace


class ARCAFDynamics(nn.Module):
    def __init__(self, fdyn: nn.Module, adyn: nn.Module, estimate_trace: bool = False):
        super().__init__()
        self.fdyn = fdyn
        self.adyn = adyn
        self.estimate_trace = estimate_trace

    def evaluate_trace(
        self, x_out: Tensor, x_in: Tensor, noise: Union[Tensor, None]
    ) -> Tuple[Tensor, Tensor]:
        if self.estimate_trace:
            trJ, J = hutchinson_trace_estimator(x_out, x_in, noise)
        else:
            trJ, J = trace(x_out, x_in, return_jacobian=True)

        return trJ, J

    def forward(
        self,
        t: Tensor,
        states: Union[List[Tensor], Tensor],
        c: Union[List[Tensor], Tensor, None] = None,
        noise: Union[Tensor, None] = None,
        **kwargs,
    ) -> List[Tensor]:

        pos, aug, *_ = states

        # estimate trace => noise is not None
        assert not self.estimate_trace or noise is not None, (
            f"Noise should be passed to dynamic forward when ",
            f"using hutchinson trace estimator.",
        )

        with torch.set_grad_enabled(True):
            pos = pos.requires_grad_(True)

            pos_dot = self.fdyn.forward(t, torch.cat([pos, aug], -1), c, **kwargs)
            aug_dot = self.adyn.forward(t, aug, c, **kwargs)

            # compute/approximate trace
            trJ, J = self.evaluate_trace(pos_dot, pos, noise=noise)

            # kinetic energy
            e_dot = torch.sum(pos_dot**2, 1)

            # frobenius jacobian
            n_dot = torch.sum(J**2, 1)

        return pos_dot, aug_dot, -trJ, -e_dot, -n_dot


class AugmentedRegularisedContinuousAmbientFlow(ODE):
    """The `AugmentedRegularisedContinuousAmbientFlow` class is a subclass of ODE that integrates the
    dynamics of a continuous flow and computes the change in log density, kinetic energy, and Frobenius
    norm of the Jacobian.
    """

    def __init__(
        self,
        function_dynamics: nn.Module,
        augment_dynamics: nn.Module,
        estimate_trace: bool = False,
        noise_distribution: Union[DistributionProtocol, None] = None,
        **kwargs,
    ):
        dynamics = ARCAFDynamics(function_dynamics, augment_dynamics)

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

    def integrate(
        self,
        pos: Tensor,
        t_span: Tensor,
        batch: Union[Tensor, None] = None,
        condition: Union[Tensor, None] = None,
        return_time_steps: bool = False,
        **kwargs,
    ):
        divergence, augmentation, kinetic_energy, norm_jacobian = torch.zeros(
            4, pos.shape[0]
        ).to(pos.device)

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

            # condition = condition[batch]

        dynamics = lambda t, x: self.dynamics.forward(
            t, x, condition, noise=epsilon, batch=batch
        )

        pos, augmentation, divergence, kinetic_energy, norm_jacobian = self.odeint(
            dynamics,
            (pos, augmentation[:, None], divergence, kinetic_energy, norm_jacobian),
            t_span,
            self.dynamics.parameters(),
        )

        if not return_time_steps:
            return pos[-1], divergence[-1], kinetic_energy[-1], norm_jacobian[-1]

        return pos, divergence, kinetic_energy, norm_jacobian


# if __name__ == "__main__":
#     import torch.nn as nn
#     import numpy as np

#     def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
#         r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`
#         Args:
#             v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
#             b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`
#         Returns:
#             Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`
#         See :class:`~rff.layers.GaussianEncoding` for more details.
#         """
#         vp = 2 * np.pi * v @ b.T
#         return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

#     class FDyn(nn.Module):
#         """ Models the dynamics of the injection to the manifold. """

#         def __init__(self):
#             super().__init__()
#             # expected format: N x (C * L)
#             # +1 for time
#             self.fc = nn.Sequential(nn.Linear(16, 512), nn.Tanh(), nn.Linear(512, 3))

#             embed_dim = 8
#             self.Wx = nn.Parameter(torch.randn(2 // 2, 3), requires_grad=False)
#             self.Wc = nn.Parameter(torch.randn(12 // 2, 1), requires_grad=False)
#             self.Wt = nn.Parameter(torch.randn(2 // 2, 1), requires_grad=False)

#         def forward(self, t, x, c, **kwargs):
#             # x_proj = x @ self.W[None, :] * 2 * np.pi
#             # TODO: this is not correct
#             #
#             x = gaussian_encoding(x, self.Wx)
#             c = gaussian_encoding(c, self.Wc)
#             t = gaussian_encoding(t.unsqueeze(0), self.Wt)
#             x = torch.concat([x, c, t.repeat([x.shape[0], 1])], -1)

#             return self.fc(x)

#     dynamics = RCAFDynamics(FDyn())
#     network = RegularisedContinuousAmbientFlow(
#         dynamics=dynamics,
#         estimate_trace=False,
#         # adjoint=True
#         adjoint=False,
#         method="rk4",
#         atol=1e-9,
#         rtol=1e-9,
#     )

#     x1 = torch.zeros(1, 3)
#     z1 = torch.ones(1, 3)
#     c1 = torch.tensor([[1]])

#     x2 = torch.zeros(1, 3)
#     z2 = torch.zeros(1, 3)
#     c2 = torch.tensor([[2]])

#     # dopri5,1e-9: convergence after 34, final: 1.8586704847748867e-14
#     # dopri5,1: convergence after 35, final: 2.3064803715490585e-14

#     dataset = [
#         (x1, z1, c1),
#         (x2, z2, c2),
#     ]

#     optimiser = torch.optim.Adam(network.parameters())

#     for i in range(200):
#         for x, z, c in dataset:
#             optimiser.zero_grad()

#             z_tilde, *_ = network.inverse(x, None, c)

#             loss = (z_tilde - z).pow(2).mean()
#             print(f"Epoch {i}: loss {loss.item()}")
#             loss.backward()
#             optimiser.step()

#     breakpoint()
