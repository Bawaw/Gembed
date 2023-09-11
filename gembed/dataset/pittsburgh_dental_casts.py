#!/usr/bin/env python3
import os
import glob
import torch

from gembed.utils.adapter import vtk_to_torch_geometric_data
from gembed.dataset.abstract_dataset import AbstractDataset
from gembed import Configuration


class PittsburghDentalCasts(AbstractDataset):
    """ Segmented livers from the medical segmentation decathlon dataset."""

    @property
    def _raw_data(self):
        import pyvista as pv

        dcm_files = glob.glob(os.path.join(self.raw_dir, "stls/*/*/*/*.stl"))

        # format [(id, Data), ...]
        raw_data = []
        for path in dcm_files:
            data = vtk_to_torch_geometric_data(pv.read(path))
            data.pos = data.pos.float()
            data["id"] = [os.path.dirname(path).split("/")[-1]]
            data["location"] = [os.path.dirname(path).split("/")[-2]]
            data["subgroup"] = [os.path.dirname(path).split("/")[-3]]

            raw_data.append(data)

        # sort files based on patient id
        raw_data.sort(key=lambda d: d["id"])

        for i, d in enumerate(raw_data):
            d.idx = i

        return raw_data

    def process(self):
        return super().process()


class PittsburghDentalCastsCurvature(AbstractDataset):
    """ Segmented livers from the medical segmentation decathlon dataset."""

    @property
    def _raw_data(self):
        import pyvista as pv
        from plyfile import PlyData

        dcm_files = glob.glob(os.path.join(self.raw_dir, "*/*/*/*.ply"))

        # format [(id, Data), ...]
        raw_data = []
        for path in dcm_files:
            data = vtk_to_torch_geometric_data(pv.read(path))
            data.pos = data.pos.float()
            data["id"] = [os.path.dirname(path).split("/")[-1]]
            data["location"] = [os.path.dirname(path).split("/")[-2]]
            data["subgroup"] = [os.path.dirname(path).split("/")[-3]]
            # TODO: fix this so we only need to read the data once
            data["x"] = torch.from_numpy(
                PlyData.read(path)["vertex"]["quality"][:, None].astype("float32")
            )

            raw_data.append(data)

        # sort files based on patient id
        raw_data.sort(key=lambda d: d["id"])

        for i, d in enumerate(raw_data):
            d.idx = i

        return raw_data

    def process(self):
        return super().process()
