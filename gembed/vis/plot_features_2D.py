# /usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from umap import UMAP


def plot_features_2D(
    Zs, Z_colors=None, umap_kwargs={}, file_name=None, Z_trajs={}, pca=True, umap=True, **kwargs
):

    assert pca or umap, "Expect at least one to be true for feature representation; pca or umap."



    latent_1 = []
    latent_2 = []
    method = []
    colors = []

    # embed data using PCA
    if pca:
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Zs.clone())

        latent_1.append(Z_pca[:, 0])
        latent_2.append(Z_pca[:, 1])
        method.append(["PCA"] * Zs.shape[0])
        if Z_colors is not None:
            colors.append(Z_colors)

    # embed data using UMAP
    if umap:
        umap = UMAP(n_components=2, random_state=42, **umap_kwargs)
        Z_umap = umap.fit_transform(Zs.clone())

        latent_1.append(Z_umap[:, 0])
        latent_2.append(Z_umap[:, 1])
        method.append(["UMAP"] * Zs.shape[0])
        if Z_colors is not None:
            colors.append(Z_colors)

    data = {
        "latent_1": np.concatenate(latent_1),
        "latent_2": np.concatenate(latent_2),
        "Method": np.concatenate(method),
        "hue" : colors if colors != [] else None
    }

    # fix order of hue and style
    fig = sns.relplot(
        data=data,
        x="latent_1",
        y="latent_2",
        col="Method",
        hue="hue" if data["hue"] is not None else None,
        facet_kws=dict(sharex=False, sharey=False),
        **kwargs
    )
    colors = sns.color_palette()

    # plot interpolation trajectories
    for i, k in enumerate(Z_trajs.keys()):
        if pca:
            Z_traj_pca = pca.transform(Z_trajs[k].clone())

            fig.axes[0][0].plot(
                Z_traj_pca[:, 0], Z_traj_pca[:, 1], marker="o", label=k, color=colors[i + 1]
            )
            fig.axes[0][0].scatter(
                Z_traj_pca[[0, -1], 0], Z_traj_pca[[0, -1], 1]
            )  # start and endpoint

            fig.axes[0][0].legend()

        # umap transform is stochastic, running one element at a time reduces this
        # https://github.com/lmcinnes/umap/issues/566
        if umap:
            # TODO: support just umap only feature plots
            assert umap and not pca, "UMAP only feature plot not supported."
            Z_traj_umap = torch.concat(
                [torch.from_numpy(umap.transform(Z[None])) for Z in Z_trajs[k].clone()]
            )

            fig.axes[0][1].plot(
                Z_traj_umap[:, 0],
                Z_traj_umap[:, 1],
                marker="o",
                label=k,
                color=colors[i + 1],
            )
            fig.axes[0][1].scatter(
                Z_traj_umap[[0, -1], 0], Z_traj_umap[[0, -1], 1]
            )  # start and endpoint

            fig.axes[0][1].legend()


    if file_name is not None:
        fig.savefig(file_name, bbox_inches="tight", dpi=300)

    plt.show()

    return fig


if __name__ == "__main__":
    import torch

    plot_features_2D(
        Zs=torch.randn(1000, 10),
        Z_trajs={"a": torch.ones(100, 10) * torch.linspace(-3, 3, 100)[:, None]},
    )
