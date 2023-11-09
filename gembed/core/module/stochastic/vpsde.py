#!/usr/bin/env python3

import torch
from gembed.core.module.stochastic import AbstractSDE
from gembed.core.distribution import MultivariateNormal


class VPSDE(AbstractSDE):
    """Variance preserving Stochastic differential equation, this is the continuous version of the DDPM model.

    Source: This code is based on https://github.com/yang-song/score_sde/blob/main/sde_lib.py
    """

    def __init__(self, beta_min, beta_max, **kwargs):
        super().__init__(**kwargs)

        print(f"VPSDE(β_min: {beta_min}, β_max: {beta_max})")

        self.beta_min = beta_min
        self.beta_max = beta_max
        # TODO: move this to constructor argument
        self.base_distribution = MultivariateNormal(
            torch.zeros(3),
            torch.eye(3, 3),
        )
        self.beta = lambda t: torch.lerp(
            torch.Tensor([self.beta_min]).to(t.device),
            torch.Tensor([self.beta_max]).to(t.device),
            t,
        )

    def drift_coefficient(self, x, t, batch=None, condition=None):
        # f(x, t)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor([None, t])
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        # -1/2 * βₜ * x
        return (-0.5 * self.beta(t)).to(x.device)[batch] * x

    def diffusion_coefficient(self, x, t, batch=None, condition=None):
        # G(x, t)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor(t)
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        # √βₜ
        result = torch.sqrt(self.beta(t)).to(t.device)
        return result[batch]

    def base_log_prob(self, z):
        return self.base_distribution.log_prob(z)

    def sample_base(self, n_samples):
        return self.base_distribution.sample(n_samples)

    def marginal_prob_params(self, x, t, batch=None, condition=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if isinstance(t, int):
            t = torch.Tensor(t)
        elif t.dim() == 1:
            t = t[None]

        assert batch.max() + 1 == t.shape[0], (
            f"Discrepancy between number of batches {batch.max() + 1}"
            f"and number of time steps {t.shape[0]}.")

        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )[batch]
        mean = x * torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))

        return mean, std


if __name__ == "__main__":
    model = VPSDE(None, 0.0, 3)

    from pyvista import examples
    from gembed.vis import plot_objects

    # 1)
    # x = torch.from_numpy(examples.download_bunny().points)
    # x = (x - x.mean(0)) / x.std(0).max()

    # N = 5
    # dt = torch.Tensor([1 / (N - 1)])

    # objects = []
    # for t in torch.linspace(0, 1, N):
    #     z = torch.randn(x.shape[0], 3)
    #     f = model.drift_coefficient(x=x, t=t)
    #     G = model.diffusion_coefficient(x, t)

    #     # dx = f(x, t) dt + g(t) dw
    #     x = x + (f*dt) + (G * z * torch.sqrt(dt))

    #     objects.append((x, None))

    # from gembed.vis import plot_objects
    # plot_objects(* objects)

    # 2)
    x0 = torch.from_numpy(examples.download_bunny().points)
    x0 = (x0 - x0.mean(0)) / x0.std(0).max()

    N = 5
    dt = torch.Tensor([1 / (N - 1)])

    objects = []
    for t in torch.linspace(0, 1, N):
        z = torch.randn(x0.shape[0], 3)

        # p_t(x)
        mean, std = model.marginal_prob_params(x0, t)

        # x ~ p_t(x)
        x = mean + std * z

        objects.append((x, None))

    from gembed.vis import plot_objects

    plot_objects(*objects)
