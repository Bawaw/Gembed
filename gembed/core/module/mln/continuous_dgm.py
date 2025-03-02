#!/usr/bin/env python3
import torch
from torch import nn
from torch_geometric.utils.sparse import dense_to_sparse

# Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x, x) ** 2
    return dist, x


# #Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x ** 2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    x = x / (x_norm * (1 + 1e-2))
    x_norm = (x ** 2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = (
        torch.arccosh(
            1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))
        )
        ** 2
    )
    return dist, x


class ContinuousDGM(nn.Module):
    """
    The Continuous Differentiable Graph Module class is a PyTorch module that learns a continuous graph based on a feature distance 
    and uses this graph for information sharing between connected nodes.
    
    Source: Code based on https://github.com/lcosmo/DGM_pytorch/blob/489b43c69af7321b93c3529edb6537fa73325e07/DGMlib/layers.py#L33."""

    def __init__(self, embed_f, input_dim, distance="euclidean"):
        super().__init__()
        # modules
        self.embed_f = embed_f

        # metric
        if distance == "euclidean":
            self.distance = pairwise_euclidean_distances
        elif distance == "hyperbolic":
            self.distance = pairwise_poincare_distances
        else:
            raise ValueError("Invalid distance argument for ContinuousDGM.")

        # module parameters
        self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, A=None):
        # node feature representation
        if A is None:
            x = self.embed_f(x)

        if A is not None:
            x = self.embed_f(x, A)

        #compute the distance matrix
        D, _ = self.distance(x)
        A = torch.sigmoid(self.temperature.abs() * (self.threshold.abs() - D))
        W = (A / A.sum(1))

        self.log("Zero neighbours", float((W < 0.01).sum().item()), batch_size=x.shape[0])
        return W @ x
