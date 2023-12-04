from torch_geometric.transforms import BaseTransform

class SwapAxes(BaseTransform):
    def __init__(
        self,
        axes=[0, 1, 2],
    ):
        self.axes = axes

    def __call__(self, data):
        data.pos = data.pos[:, self.axes]
        return data
