import numpy as np
import pyvista as pv
import torch
from torch_geometric.transforms import BaseTransform

from gembed.utils import adapter


class SegmentMesh(BaseTransform):
    def __init__(self, threshold=0, fill_holes=True, fill_hole_area=1, remove_x=True):
        self.threshold = threshold
        self.fill_holes = fill_holes
        self.fill_hole_area = fill_hole_area
        self.remove_x = remove_x

    def __call__(self, data):
        pv_data = adapter.torch_geomtric_data_to_vtk(data.pos, data.face)

        # clip mesh
        pv_data["segmentation"] = data.x
        pv_data = pv_data.clip_scalar("segmentation", invert=False)

        # fill potential holes
        if self.fill_holes:
            pv_data = pv_data.fill_holes(self.fill_hole_area)

        # plot result
        # pv_data.plot()

        # assign new vertices and faces to original mesh
        data.face = (
            torch.from_numpy(np.copy(pv_data.faces)).reshape(-1, 4).permute(1, 0)[1:]
        )
        data.pos = torch.from_numpy(np.copy(pv_data.points))

        # remove the segmentation mask
        if self.remove_x:
            del data.x

        return data
