#!/usr/bin/env python3

import torch
import numpy as np
from scipy.interpolate import interp1d


def discrete_geodesic(
    X0,
    X1,
    f_local_metric=lambda x, y: (x - y).pow(2).sum(-1),
    n_cps=5,
    return_energy=False,
    n_iters=100,
    verbose=False,
    patience=20,
    decimals=6,
):
    """
    Computed the discrete geodesic curve between two points $X_1$ and $X_2$ based on a discretised curve consisting of n_cps nodes. It approximates the geodesic curve by iteratively optimizing the positions of intermediate points along the curve. If cps is None, the model instantiates the curve as a linear path.

    Parameters
        X0: The starting point of the geodesic curve.
        X1: The ending point of the geodesic curve.
        f_local_metric (optional): A function that computes the local metric (distance) between two points. Default is the squared Euclidean distance.
        cps (optional): Initial positions of the control points along the curve. If not provided, linear interpolation is used to generate the initial positions.
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
    assert n_cps > 2, f"Expected atleast 3 control points but got {n_cps}."

    # Note: we detach X0 and X1 here and create an seperate graph
    C0, C1 = X0.detach(), X1.detach()

    # if not initial curve, use linear interpolation of points
    Cs = torch.concat(
        [
            torch.lerp(input=C0, end=C1, weight=t)
            for t in torch.linspace(0, 1, n_cps).to(X1.device)
        ]
    )[1:-1]

    # Δt
    delta_t = 1 / (n_cps - 1)

    with torch.set_grad_enabled(True):

        # SETUP OPTIMISATION PARAMETERS
        counter = 0
        min_energy = float("inf")

        # optimise the control points between start and end-point
        Cs = Cs.requires_grad_(True)
        optimiser = torch.optim.Adam([Cs], lr=0.1)

        # Euler integration scheme
        for i in range(n_iters):
            optimiser.zero_grad()

            # first and last step remain fixed
            Xs = torch.concat([C0, Cs, C1])

            # compute distance between consecutive pairs
            consec_dist = f_local_metric(Xs[:-1], Xs[1:]) / delta_t
            assert consec_dist.shape == torch.Size([n_cps - 1])

            # E(γ) = 0.5 * (∫g(γ˙(t), γ˙(t))) dt
            energy = 0.5 * consec_dist.sum()

            if verbose:
                print(f"Discrete Geodesic, Epoch: {i}, Energy: {energy:.6f}")

            # EARLY STOPPING
            energy_rounded = energy.round(decimals=decimals)
            if energy_rounded < min_energy:
                min_energy = energy_rounded
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

            energy.backward()
            optimiser.step()

        # final intermediate control points
        Cs = Cs.detach()

    # Note: Reattach controlpoints to computational graph
    geodesic = torch.concat([X0, Cs, X1])
    energy = 0.5 * (f_local_metric(geodesic[:-1], geodesic[1:]) / delta_t).sum()

    if not return_energy:
        return geodesic

    return geodesic, energy


def continuous_geodesic(
    X0,
    X1,
    f_metric_tensor,
    init_steps=None,
    n_cps=5,
    return_energy=False,
    return_spline=False,
    n_iters=100,
    batch_size=1,
    verbose=False,
    patience=20,
    decimals=6,
):
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

    # tensor should be represented in batch format of size 1 [1 x n]
    assert (X0.shape[0] == 1) and (X1.shape[0] == 1)

    # if not initial curve, use linear interpolation of points
    C0, C1 = X0.detach(), X1.detach()

    # Note: we detach X0 and X1 here and create an seperate graph
    ts = torch.linspace(0, 1, n_cps).to(X1.device)
    Cs = torch.concat([torch.lerp(input=C0, end=C1, weight=t) for t in ts])[1:-1]

    with torch.set_grad_enabled(True):

        # SETUP OPTIMISATION PARAMETERS
        patience = 20
        decimals = 6
        counter = 0
        min_loss = float("inf")

        # optimise the control points between start and end-point
        Cs = Cs.requires_grad_(True)
        optimiser = torch.optim.Adam([Cs], lr=0.1)

        print("Warning fixed controlpoints")

        # Euler integration scheme
        for i in range(n_iters):
            optimiser.zero_grad()

            # create spline
            spline = NaturalCubicSpline(
                natural_cubic_spline_coeffs(ts, torch.concat([C0, Cs, C1]))
            )

            # sample spline
            t_samples = torch.rand(batch_size).to(C0.device)
            X_samples = spline.evaluate(t_samples)
            X_dot_samples = spline.derivative(t_samples)

            # compute riemannian dot product
            G_X = f_metric_tensor(X_samples)

            # evaluate average dot products v^TGv
            loss = torch.einsum(
                "ni,nij,nj->n", X_dot_samples, G_X, X_dot_samples
            ).mean()

            if verbose:
                print(f"Continuous Geodesic, Epoch: {i}, loss: {loss:.6f}")

            # # EARLY STOPPING
            loss_rounded = loss.round(decimals=decimals)
            if loss_rounded < min_loss:
                min_loss = loss_rounded
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

            loss.backward()
            optimiser.step()

        # final intermediate control points
        Cs = Cs.detach()

    spline = NaturalCubicSpline(
        natural_cubic_spline_coeffs(ts, torch.concat([C0, Cs, C1]))
    )
    breakpoint()
    t_samples = torch.linspace(0, 1, n_cps).to(C0)
    X_samples = spline.evaluate(t_samples)

    if not return_energy and not return_spline:
        return X_samples

    G_X = f_metric_tensor(X_samples)

    delta_X_samples = X_samples[1:] - X_samples[:-1]

    # Δt
    delta_t = 1 / (n_cps - 1)
    # https://en.wikipedia.org/wiki/Geodesic
    energy = (
        0.5
        * (
            torch.einsum("ni,nij,nj->n", delta_X_samples, G_X[:-1], delta_X_samples)
            / delta_t
        ).sum()
    )

    result = (X_samples,)
    if return_energy:
        result += (energy,)

    if return_spline:
        result += (spline,)

    return result

    # geodesic = torch.concat([X0, Cs, X1])
    # energy = 0.5 * (f_local_metric(geodesic[:-1], geodesic[1:]) / delta_t).sum()

    # if not return_energy:
    #     return geodesic

    # return geodesic, energy


# if __name__ == "__main__":
#     X1 = torch.tensor([[0.0, 0.0]])
#     X2 = torch.tensor([[1.0, 1.0]])

#     X1 = X1.requires_grad_(True)
#     X2 = X2.requires_grad_(True)

#     # DISCRETE GEODESIC
#     # custom metric
#     f_local_metric = lambda x, y: (x - y).pow(2).sum(-1)

#     # # Compute the geodesic curve
#     geodesic, energy = discrete_geodesic(X1, X2, return_energy=True)

#     print(f"Discrete geodesic: {geodesic}")
#     print(
#         f"Energy of the curve : {energy} is equal 1/2 * global distance {0.5*f_local_metric(X1,X2).item()} "
#     )

#     # CONTINUOUS GEODESIC
#     geodesic, energy = continuous_geodesic(
#         X1, X2, riemannian_metric_G, return_energy=True
#     )
#     print(f"Continuous geodesic: {geodesic}")
#     print(
#         f"Energy of the curve : {energy} is equal 1/2 * global distance {0.5*f_local_metric(X1,X2).item()} "
#     )
