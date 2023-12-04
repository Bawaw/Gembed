#!/usr/bin/env python3

import sys

import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects


def frechet_mean(
    Xs,
    init_mean=None,
    f_global_metric=lambda x, y: (x - y).pow(2).sum(-1),
    n_iters=int(1e4),
    verbose=False,
    patience = 20,
    decimals = 6,
):
    r"""
    This function computes the Frechet mean of a set of data points Xs. The Fréchet mean represents a central or average point in the dataset by minimizing the average distance between the points and the mean.

    Parameters
        Xs: Tensor containing the input data points.
        init_mean (optional): Initial mean point for the Fréchet mean calculation. If not provided, the first data point from the Xs tensor is used.
        f_global_metric (optional): A function that computes the global metric (distance) between two points. Default is the squared Euclidean distance.
        n_iters (optional): The number of iterations for the optimization process. Default is 100.

    Returns
        X_hat: The Fréchet mean or the central point obtained after optimization.

    $$
      \bar{X} = \underset{\bar{X}}{\arg\min} \frac{1}{n}\sum_{i=1}^{n} g(X, X_i)
    $$

    """

    # if not initial mean, use mean datapoint
    if init_mean is None:
        init_mean = Xs.mean(0, keepdim=True)

    with torch.set_grad_enabled(True):
        # SETUP OPTIMISATION PARAMETERS
        counter = 0
        min_dist = float("inf")

        # optimise for mean
        X_hat = init_mean.requires_grad_(True)
        optimiser = torch.optim.Adam([X_hat])

        for i in range(n_iters):
            optimiser.zero_grad()

            distance = torch.stack([f_global_metric(X, X_hat) for X in Xs.unsqueeze(1)])
            average_distance = distance.mean()
            if verbose:
                print(f"Frechet Mean, Epoch: {i}, Average distance: {average_distance:.4f}")

            # EARLY STOPPING
            average_distance_rounded = average_distance.round(decimals=decimals)
            if average_distance_rounded < min_dist:
                min_dist = average_distance_rounded
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

            average_distance.backward()
            optimiser.step()

    return X_hat.detach()

if __name__ == "__main__":
    Zs = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    mean_point = frechet_mean(Zs)

    print("Fréchet Mean:", mean_point)
