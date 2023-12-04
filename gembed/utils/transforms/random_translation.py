import torch
from torch_geometric.transforms import BaseTransform

class RandomTranslation(BaseTransform):
    def __init__(self, sigma=0.2):
        self.sigma = sigma

    def __call__(self, data):
        b_transform = self.sigma * torch.randn(3)

        data.pos = data.pos + b_transform

        return data
