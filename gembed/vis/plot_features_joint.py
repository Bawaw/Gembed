# /usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from umap import UMAP


def plot_features_joint(Zs, Z_colors=None, file_name=None, Z_trajs={}, **kwargs):
    # embed data using PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Zs.clone())

    data = {
        "pc_1": Z_pca[:, 0],
        "pc_2": Z_pca[:, 1],
    }

    if Z_colors is not None:
        data["hue"] = Z_colors
    else:
        data["hue"] = None

    # fix order of hue and style
    fig = sns.jointplot(data=data, x="pc_1", y="pc_2", hue="hue", **kwargs)
    colors = sns.color_palette()
    for i, k in enumerate(Z_trajs.keys()):
        Z_traj_pca = pca.transform(Z_trajs[k].clone())

        fig.ax_joint.plot(
            Z_traj_pca[:, 0], Z_traj_pca[:, 1], marker="o", label=k, color=colors[i + 1]
        )
        fig.ax_joint.scatter(
            Z_traj_pca[[0, -1], 0], Z_traj_pca[[0, -1], 1]
        )  # start and endpoint
        fig.ax_joint.legend()

    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight")

    plt.show()

    return fig


if __name__ == "__main__":
    import torch

    plot_features_2D(
        Zs=torch.randn(1000, 10),
        Z_trajs={"a": torch.ones(100, 10) * torch.linspace(-3, 3, 100)[:, None]},
    )
