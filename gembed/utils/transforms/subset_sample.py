import numpy as np
from torch_geometric.transforms import BaseTransform

class SubsetSample(BaseTransform):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __call__(self, data):
        indices = np.random.choice(data.pos.shape[0], self.n_samples)

        data.pos = data.pos[indices]

        return data
