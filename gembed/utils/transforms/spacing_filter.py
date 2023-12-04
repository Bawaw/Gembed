from torch_geometric.transforms import BaseTransform

class SpacingFilter(BaseTransform):
    import SimpleITK as sitk

    def __init__(self, spacing):
        self.spacing = spacing

    def __call__(self, data):
        mask = data["vol"]

        return (np.array(data["vol"].GetSpacing()) <= self.spacing).all()
