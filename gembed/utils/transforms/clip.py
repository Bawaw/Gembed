from torch_geometric.transforms import BaseTransform

class Clip(BaseTransform):
    def __init__(self, threshold=1.2):
        self.threshold = threshold

    def __call__(self, data):
        data.pos = data.pos[data.pos[:, 2] > self.threshold]
        return data
