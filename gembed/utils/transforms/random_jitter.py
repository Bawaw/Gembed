from torch_geometric.transforms import BaseTransform

class RandomJitter(BaseTransform):
    def __init__(self, scale, dim=3, distribution="normal"):
        if distribution == "normal":
            self.distribution = torch.distributions.Normal(
                torch.tensor([0.0] * dim), torch.tensor([scale] * dim)
            )
        elif distribution == "laplace":
            self.distribution = torch.distributions.Laplace(
                torch.tensor([0.0] * dim), torch.tensor([scale] * dim)
            )
        else:
            raise ValueError("Only normal/laplace distributions supported for Jitter.")

    def __call__(self, data):
        data.pos = data.pos + self.distribution.sample(data.pos.shape[:1])
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.distribution)
