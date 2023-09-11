#!/usr/bin/env python3
import os
import glob

from gembed.utils.adapter import vtk_to_torch_geometric_data
from gembed.dataset.abstract_dataset import AbstractDataset
from gembed import Configuration


class ABCDBrain(AbstractDataset):
    """ Segmented livers from the medical segmentation decathlon dataset."""

    @property
    def _raw_data(self):
        import pyvista as pv

        dcm_files = glob.glob(os.path.join(self.raw_dir, "*/surf/lh_pial.vtk"))

        # format [(id, Data), ...]
        raw_data = []
        for path in dcm_files:
            data = vtk_to_torch_geometric_data(pv.read(path))
            data.pos = data.pos.float()
            data["id"] = [os.path.dirname(path).split("/")[-2]]

            raw_data.append(data)

        # sort files based on patient id
        raw_data.sort(key=lambda d: d["id"])

        return raw_data

    def process(self):
        return super().process()
