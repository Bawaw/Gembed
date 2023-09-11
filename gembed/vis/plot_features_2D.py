# /usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from umap import UMAP


def plot_features_2D(Zs, Z_colors=None, umap_kwargs=None, file_name=None, **kwargs):
    # embed data using PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Zs.clone())

    # embed data using UMAP
    pca = UMAP(n_components=2, random_state=42, **umap_kwargs)
    Z_umap = pca.fit_transform(Zs.clone())

    data = {
        "latent_1": np.concatenate(
            [
                Z_pca[:, 0],
                Z_umap[:, 0],
            ]
        ),
        "latent_2": np.concatenate(
            [
                Z_pca[:, 1],
                Z_umap[:, 1],
            ]
        ),
        "Method": np.concatenate(
            [
                ["PCA"] * Zs.shape[0],
                ["UMAP"] * Zs.shape[0],
            ]
        ),
    }

    if Z_colors is not None:
        data["hue"] = torch.cat([Z_colors, Z_colors])

    # fix order of hue and style
    fig = sns.relplot(
        data=data,
        x="latent_1",
        y="latent_2",
        col="Method",
        hue="hue",
        facet_kws=dict(sharex=False, sharey=False),
        **kwargs
    )

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")
    plt.show()
