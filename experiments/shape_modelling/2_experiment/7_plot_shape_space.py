#!/usr/bin/env python3

import os
import torch
import lightning as pl
import matplotlib.pyplot as plt
import torch_geometric.transforms as tgt

from gembed.vis import plot_features_2D
from helper import load_experiment, pathcat

def plot_result(Zs, file_path):
    # plot embeddings
    fig = plot_features_2D(
        Zs,
        file_name=file_path,
        umap_kwargs={"metric": "euclidean", "min_dist": 0.9, "spread": 0.9},
        umap=False,
    )

def plot_latent_space(model, Zs, file_path):
    plot_result(Zs.cpu(), file_path)

def plot_normalised_space(model, Zs, file_path):
    Zs_norm = torch.concat([model.ltn.inverse(Z) for Z in Zs.unsqueeze(1)]).cpu()

    plot_result(Zs_norm, file_path)

def plot_metricised_space(model, Zs, file_path):
    Zs_metric = torch.concat([model.mtn.inverse(Z) for Z in Zs.unsqueeze(1)]).cpu()

    plot_result(Zs_metric, file_path)

def run(
    model,
    T_sample,
    dataset,
    device="cpu",
    umap_metric="euclidean",
    file_path=None,
):
    pl.seed_everything(42, workers=True)

    # data transform
    T = tgt.Compose(
        [
            T_sample,
            tgt.NormalizeScale(),
        ]
    )

    # Compute latent embeddings
    Zs = torch.concat(
        [
            model.inverse(
                T(d.clone()).pos.to(device), None, apply_ltn=False, apply_stn=True
            )
            for d in dataset
        ]
    )

    # plot
    if file_path is not None:
        os.makedirs(file_path, exist_ok=True)
    plot_latent_space(model, Zs, pathcat(file_path, "latent_space.svg"))
    if model.ltn is not None:
        plot_normalised_space(model, Zs, pathcat(file_path, "normalised_space.svg"))
    if model.mtn is not None:
        plot_metricised_space(model, Zs, pathcat(file_path, "metricised_space.svg"))

def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, train, device=device, file_path=pathcat(file_path, "train"))

if __name__ == "__main__":
    main()