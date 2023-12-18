import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class BinaryMask2Volume(BaseTransform):
    import SimpleITK as sitk

    def __init__(
        self,
        remove_pixel_information=True,
    ):
        self.remove_pixel_information = remove_pixel_information

    def __call__(self, data):
        mask = data["vol"]

        mask_arr = self.sitk.GetArrayFromImage(mask)
        indices = np.stack(mask_arr.nonzero(), -1).astype(int)

        new_data = Data(
            pos=torch.FloatTensor(
                [mask.TransformIndexToPhysicalPoint(i.tolist()) for i in indices]
            )
        )

        # copy other data attributes
        for k in data:
            if self.remove_pixel_information and k == "vol":
                continue

            new_data[k] = data[k]

        return new_data
