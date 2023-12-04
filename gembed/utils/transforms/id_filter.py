from torch_geometric.transforms import BaseTransform

class IDFilter(BaseTransform):
    def __init__(self, ids):
        self.ids = ids

    def __call__(self, data):
        return not (data["id"][0] in self.ids)


