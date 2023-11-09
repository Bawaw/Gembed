#!/usr/bin/env python3

import torch
from umap import UMAP
from torch_geometric.transforms import (BaseTransform, Compose, NormalizeScale,
                                        SamplePoints, Center)

def plot_shape_space(model, dataset, n_samples=10):
    # augmentation strategies
    augment_transform = Compose([SamplePoints(2048)])
    #augment_transform = Compose([SamplePoints(2048), Clip(), RandomRotation(), Center()])

    # dataset
    n_samples = max(n_samples, len(dataset))
    dataset = [dataset[i].clone() for i in range(n_samples)]

    # get embedding for each augmented datapoint
    embeddings = [(i, model.inverse(augment_transform(dataset[i].clone()).pos, None))
                  for i in range(len(dataset)) for _ in range(n_augmentation_samples)]
    labels, embeddings = zip(*embeddings)
    labels, embeddings = torch.Tensor(labels), torch.concat(embeddings)

    # reduce dimension
    print("Warning: using different parametrisation for UMAP.")
    umap = UMAP(n_components=2, random_state=42)
    #umap = UMAP(n_components=2, n_neighbors=20, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)

    # only focus on the first 9 datapoints
    labels[labels > 9] = 0

    # plot results
    sns.scatterplot(data={
        "z_1": embeddings_2d[:, 0],
        "z_2": embeddings_2d[:, 1]
    }, x="z_1", y="z_2", hue=labels, palette="Set3")

    plt.show()
