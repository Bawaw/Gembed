#!/usr/bin/env python3

import torch
import numpy as np
from scipy.interpolate import interp1d


def discrete_geodesic(
    X0,
    X1,
    f_local_metric=lambda x, y: (x - y).pow(2).sum(-1),
    init_steps=None,
    n_cps=6,
    return_energy=False,
    n_iters=100,
    verbose=False
):
    """
    Computed the discrete geodesic curve between two points $X_1$ and $X_2$ based on a discretised curve consisting of n_cps nodes. It approximates the geodesic curve by iteratively optimizing the positions of intermediate points along the curve. If init_steps is None, the model instantiates the curve as a linear path.

    Parameters
        X0: The starting point of the geodesic curve.
        X1: The ending point of the geodesic curve.
        f_local_metric (optional): A function that computes the local metric (distance) between two points. Default is the squared Euclidean distance.
        init_steps (optional): Initial positions of the control points along the curve. If not provided, linear interpolation is used to generate the initial positions.
        n_cps (optional): The number of discrete steps (control points) (including X1 and X2) on the geodesic curve. Default is 6.
        return_energy (optional): A boolean flag indicating whether to return the energy of the curve. Default is False.
        n_iters (optional): The number of iterations for the optimization process. Default is 10.

    Returns

    If return_energy is False:
        geodesic: The tensor containing the positions of the control points along the geodesic curve.

    If return_energy is True:
        geodesic: The tensor containing the positions of the control points along the geodesic curve.
        energy: The energy of the curve.

    $$
        E(x_1, x_2) = \frac{1}{2} \sum_{i=0}^{T} \frac{1}{\delta{t}} g(x_{t+1}, x_{t})
    $$

    """

    # tensor should be represented in batch format of size 1 [1 x n]
    assert (X0.shape[0] == 1) and (X1.shape[0] == 1)

    # if not initial curve, use linear interpolation of points
    with torch.set_grad_enabled(False):
        if init_steps is None:
            init_steps = torch.concat(
                [
                    torch.lerp(input=X0, end=X1, weight=t)
                    for t in torch.linspace(0, 1, n_cps).to(X1.device)
                ]
            )
        else:
            assert (
                init_steps.shape(0) == n_cps
            ), "Size of init_steps and n_cps does not match."

        # Δt
        delta_t = 1 / (n_cps - 1)

        with torch.set_grad_enabled(True):

            # SETUP OPTIMISATION PARAMETERS
            patience = 10
            delta = 1e-2
            counter = 0
            min_energy = float('inf')

            # optimise the control points between start and end-point
            control_points = init_steps[1:-1].requires_grad_(True)
            optimiser = torch.optim.Adam([control_points])

            # Euler integration scheme
            for i in range(n_iters):
                optimiser.zero_grad()

                # first and last step remain fixed
                Xs = torch.concat(
                    [
                        init_steps[:1].requires_grad_(False),
                        control_points,
                        init_steps[-1:].requires_grad_(False),
                    ]
                )

                # compute distance between consecutive pairs
                consec_dist = f_local_metric(Xs[:-1], Xs[1:]) / delta_t
                assert consec_dist.shape == torch.Size([n_cps-1])

                # E(γ) = 0.5 * (∫g(γ˙(t), γ˙(t))) dt
                energy = 0.5 * consec_dist.sum()

                if verbose:
                    print(f"Discrete Geodesic, Epoch: {i}, Energy: {energy:.2f}")

                # when all consecutive distances are the same we converged
                if torch.isclose(consec_dist[None], consec_dist[:, None]).all():
                    break

                # EARLY STOPPING
                if energy < min_energy:
                    min_energy = energy
                    counter  = 0
                elif energy > (min_energy + delta):
                    counter += 1
                if counter >= patience:
                    break

                energy.backward()
                optimiser.step()

    geodesic = torch.concat(
        [
            X0,
            control_points,
            X1,
        ]
    )
    energy = 0.5 * (f_local_metric(geodesic[:-1], geodesic[1:]) / delta_t).sum()

    if not return_energy:
        return geodesic

    return geodesic, energy


# if __name__ == "__main__":
#     X1 = torch.tensor([0.0, 0.0])
#     X2 = torch.tensor([1.0, 1.0])

#     # custom metric
#     f_local_metric = lambda x, y: (x - y).pow(2).sum(-1)

#     # Compute the geodesic curve
#     geodesic, energy = discrete_geodesic(X1, X2, return_energy=True)

#     print(f"Geodesic: {geodesic}")
#     print(
#         f"Energy of the curve : {energy} is equal 1/2 * global distance {0.5*f_local_metric(X1,X2)} "
#     )
