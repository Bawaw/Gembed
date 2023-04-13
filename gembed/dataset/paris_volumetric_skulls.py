#!/usr/bin/env python3
import glob
import math
import os
import warnings

import numpy as np

import pyvista as pv
import torch
from gembed import Configuration

# from gembed.vis.plotter import Plotter
from gembed.models import PCA
from gembed.registration import AffineRegistration
from pytorch_lightning.core.datamodule import LightningDataModule
from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url


class ParisVolumetricSkulls(InMemoryDataset, LightningDataModule):
    """ original CT scans of the Paris dataset. """

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        isotropic_scaling=True,
        scale=None
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

        self.scale = scale

        super().__init__(root, transform, pre_transform, pre_filter)

        print(f"Loading dataset: {self}")
        self.data, self.slices, scale = torch.load(self.processed_paths[0])

        if self.scale is None:
            self.scale = scale

        assert self.scale == scale, "Data scale does not match scale of constructor argument."


    @property
    def subdir(self) -> str:
        return "paris_volumetric_skulls"

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
    def _raw_data(self):
        import SimpleITK as sitk

        dcm_files = glob.glob(os.path.join(self.raw_dir, "*/original.dcm"))

        # format [(id, Data), ...]
        raw_data = []
        for path in dcm_files:
            data = {
                "vol": sitk.ReadImage(path, sitk.sitkFloat32, imageIO="GDCMImageIO")
            }
            data["id"] = [int(path.split("/")[-2])]

            raw_data.append(data)

        # sort files based on patient id
        raw_data.sort(key=lambda d: d["id"])

        return raw_data

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # 1) get the in correspondence data
        data_list = self._raw_data

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
            # We /2 because we scale to [-1, 1]
            if self.scale is None:
                self.scale = torch.tensor(
                    max([d.pos.abs().max().item() for d in data_list])
                ) / 2

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


# if __name__ == "__main__":
#     from torch_geometric.data import Data
#     from torch_geometric.transforms import BaseTransform, Center, Compose

#     class ExtractSurfacePCByThreshold(BaseTransform):
#         import SimpleITK as sitk

#         def __init__(self, threshold=1500):
#             self.threshold = threshold

#         def __call__(self, data):

#             img = data["vol"]
#             img_arr = self.sitk.GetArrayFromImage(img)

#             indices = np.stack((img_arr > 1500).nonzero(), -1).astype(int)

#             new_data = Data(
#                 pos=torch.FloatTensor(
#                     [img.TransformIndexToPhysicalPoint(i.tolist()) for i in indices]
#                 )
#             )

#             return new_data

#     class SubsetSample(BaseTransform):
#         def __init__(self, n_samples):
#             self.n_samples = n_samples

#         def __call__(self, data):
#             indices = np.random.choice(data.pos.shape[0], self.n_samples)

#             data.pos = data.pos[indices]

#             return data

#     dataset = ParisVolumetricSkulls(
#         pre_transform=ExtractSurfacePCByThreshold(),
#         transform=Compose([Center(), SubsetSample(5012)]),
#     )

#     from gembed.vis import plot_objects

#     for d in dataset:
#         plot_objects((d, d.pos[:, 0]))
