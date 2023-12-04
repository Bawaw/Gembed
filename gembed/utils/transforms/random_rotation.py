import torch
from torch_geometric.transforms import BaseTransform

class RandomRotation(BaseTransform):
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, data):
        alpha, beta, gamma = self.sigma * torch.randn(3)

        R = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(beta) * torch.cos(gamma),
                        torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma)
                        - torch.cos(alpha) * torch.sin(gamma),
                        torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)
                        + torch.sin(alpha) * torch.sin(gamma),
                    ],
                    -1,
                ),
                torch.stack(
                    [
                        torch.cos(beta) * torch.sin(gamma),
                        torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma)
                        + torch.cos(alpha) * torch.cos(gamma),
                        torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)
                        - torch.sin(alpha) * torch.cos(gamma),
                    ],
                    -1,
                ),
                torch.stack(
                    [
                        -torch.sin(beta),
                        torch.sin(alpha) * torch.cos(beta),
                        torch.cos(alpha) * torch.cos(beta),
                    ],
                    -1,
                ),
            ],
            -1,
        )

        # [N x 3] x [3 x 3] = [N x 3]
        data.pos = data.pos @ R

        return data
