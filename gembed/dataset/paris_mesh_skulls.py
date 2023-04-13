#!/usr/bin/env python3
import glob
import os
import math

import numpy as np

import torch
from gembed import Configuration
from gembed.io.ply import read_ply
import warnings

# from gembed.vis.plotter import Plotter
from gembed.models import PCA
from gembed.registration import AffineRegistration
from pytorch_lightning.core.datamodule import LightningDataModule
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url


class CleanParisMeshSkulls(InMemoryDataset, LightningDataModule):
    """ Skull surfaces extracted from the Paris dataset. """

    def __init__(
        self, root=None, transform=None, pre_transform=None, isotropic_scaling=True
    ):
        """
        Parameters
        ----------
        root : str
            Root folder
        n_samples : int, optional
            Number of generative samples
        n_components : int, optional
            Number of components used to generate data
        seed : int, optional
            Seed used to generate data
        """

        if root is None:
            path = Configuration()["Paths"]["DATA_DIR"]
            root = os.path.join(path, self.subdir)

        self.isotropic_scaling = isotropic_scaling
        self.scale = None

        super().__init__(root, transform, pre_transform)

        print(f"Loading dataset: {self}")
        self.data, self.slices, self.scale = torch.load(self.processed_paths[0])

    @property
    def subdir(self) -> str:
        return "paris_mesh_skulls"

    def scale_isotropically(self, data, invert=False):
        assert (
            self.scale is not None
        ), "Scale is not known, is constructor argument isotropic_scaling set to True?"

        if invert:
            data.pos = data.pos * self.scale

        else:
            data.pos = data.pos / self.scale

        return data

    @property
    def _raw_in_correspondence_data(self):
        ply_files = glob.glob(os.path.join(self.raw_dir, "*/bone_template.ply"))

        # format [(id, Data), ...]
        mesh_data = []
        for path in ply_files:
            data = read_ply(path)
            data["id"] = [int(path.split("/")[-2])]

            mesh_data.append(data)

        # sort files based on patient id
        mesh_data.sort(key=lambda d: d.id)

        return mesh_data

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # 1) get the in correspondence data
        data_list = self._raw_in_correspondence_data

        # 2) preprocess the data
        data_list = [d for d in data_list if d is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 2.1) store intermeditate results
        path = os.path.join(self.processed_dir, "step_2_1")
        os.makedirs(path, exist_ok=True)
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(path, self.processed_file_names[0]))

        # 3) standardise the data
        if self.isotropic_scaling:
            # find the maximum boundary of the data samples
            self.scale = torch.tensor(
                max([(d.pos.abs().max()).item() for d in data_list])
            )

            # isotropically scale the entire dataset with the same factor
            data_list = [self.scale_isotropically(d) for d in data_list]

        # PLOT RESULTS #
        # from gembed.vis.plotter import Plotter
        # import pyvista as pv

        # _, ind = torch.sort(data_list[0].pos[:, 0])
        # scalars = torch.linspace(0, 1, len(ind))
        # scalars[ind] = scalars.clone()

        # for data in data_list:
        #     plotter = Plotter()
        #     plotter.add_generic(data, scalars=scalars)
        #     plotter.show_bounds()
        #     plotter.camera_position = "xz"
        #     plotter.show()
        # exit()
        # PLOT RESULTS #

        # 4) Store data
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.scale), self.processed_paths[0])
