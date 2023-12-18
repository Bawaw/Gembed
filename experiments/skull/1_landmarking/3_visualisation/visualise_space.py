#!/usr/bin/env python3

import os


import lightning as pl
import torch
import torch_geometric.transforms as tgt
from gembed.core.optim import gradient_ascent
from gembed.vis import plot_features_joint
from transform import SubsetSample


def embed_latent_space(model, Xs, device):
    # Compute embeddings
    return torch.concat(
        [model.inverse(X, None, apply_ltn=False, apply_stn=True) for X in Xs]
    ).cpu()


def embed_sample_space(model, Xs, device):
    # Compute embeddings
    return torch.concat(
        [model.inverse(X, None, apply_ltn=True, apply_stn=True) for X in Xs]
    ).cpu()


def embed_metric_space(model, Xs, device):
    # Compute embeddings
    Zs_latent = torch.concat(
        [model.inverse(X, None, apply_ltn=False, apply_stn=True) for X in Xs]
    )

    Zs_metric = model.mtn.inverse(Zs_latent).cpu()

    return Zs_metric


def vis_spaces(
    model,
    dataset,
    idx_start=None,
    idx_end=None,
    device="cpu",
):
    pl.seed_everything(42, workers=True)

    f_sample_points = (
        tgt.SamplePoints(8192) if hasattr(dataset[0], "face") else SubsetSample(8192)
    )

    # data transform
    T = tgt.Compose(
        [
            f_sample_points,
            tgt.NormalizeScale(),
        ]
    )

    # setup model
    model = model.to(device)
    Xs = [T(d.clone()).pos.to(device) for d in dataset]

    # project to respective spaces
    Zs_latent = embed_latent_space(model, Xs, device)
    Zs_sample = embed_sample_space(model, Xs, device)
    Zs_metric = embed_metric_space(model, Xs, device)

    if idx_start is not None and idx_end is not None:
        Z0_metric, Z1_metric = Zs_metric[idx_start, None], Zs_metric[idx_end, None]

        Zs_metric_t = torch.lerp(
            input=Z0_metric.to(device),
            end=Z1_metric.to(device),
            weight=torch.linspace(0, 1, 6)[:, None].to(device),
        )

        Zs_latent_t = model.mtn.forward(Zs_metric_t)

        Z_metric_trajs = {"Linear Interpolation": Zs_metric_t.cpu()}
        Z_latent_trajs = {"Geodesic Interpolation": Zs_latent_t.cpu()}
    else:
        Z_metric_trajs, Z_latent_trajs = {}, {}

    # plot spaces
    fig = plot_features_joint(Zs_latent, Z_trajs=Z_latent_trajs)
    fig = plot_features_joint(Zs_sample)
    fig = plot_features_joint(Zs_metric, Z_trajs=Z_metric_trajs)
