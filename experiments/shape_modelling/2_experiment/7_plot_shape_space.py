#!/usr/bin/env python3

import os
import torch
import pytorch_lightning as pl
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

def plot_latent_space(Zs, file_path):
    plot_result(Zs.cpu(), file_path)

def plot_normalised_space(Zs, file_path):
    Zs_norm = torch.concat([model.ltn.inverse(Z) for Z in Zs.unsqueeze(1)]).cpu()

    plot_result(Zs_norm, file_path)

def plot_metricised_space(Zs, file_path):
    Zs_metric = torch.concat([model.mtn.inverse(Z) for Z in Zs.unsqueeze(1)]).cpu()

    plot_result(Zs_metric, file_path)

def main(
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
    plot_latent_space(Zs, pathcat(file_path, "latent_space.svg"))
    #plot_normalised_space(Zs, pathcat(file_path, "normalised_space.svg"))
    plot_metricised_space(Zs, pathcat(file_path, "metricised_space.svg"))

if __name__ == "__main__":
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        main(model, T_sample, train + valid + test, device=device, file_path=pathcat(file_path, "complete_dataset"))
