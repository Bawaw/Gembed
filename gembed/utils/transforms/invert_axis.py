from torch_geometric.transforms import BaseTransform

class InvertAxis(BaseTransform):
    def __init__(
        self,
        axis=1,
    ):
        self.axis = axis

    def __call__(self, data):
        data.pos[:, self.axis] = -data.pos[:, self.axis]
        return data
