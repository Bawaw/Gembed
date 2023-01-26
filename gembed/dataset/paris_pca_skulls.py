#!/usr/bin/env python3
import glob
import os

import numpy as np

import torch
from gembed import Configuration
from gembed.io.ply import read_ply
#from gembed.vis.plotter import Plotter
from gembed.models import PCA
from gembed.registration import AffineRegistration
from pytorch_lightning.core.datamodule import LightningDataModule
from torch_geometric.data import (Data, DataLoader, InMemoryDataset,
                                  download_url)


class LowResParisPCASkulls(InMemoryDataset, LightningDataModule):
    """ Skulls generated using a SSM on the Paris dataset skulls. """

    def __init__(
            self, root=None, n_samples=10, n_components=28, seed=10,
            transform=None, pre_transform=None, affine_align=True,
            save_intermediate_steps=True, sample_type='rand'):
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
            path = Configuration()['Paths']['DATA_DIR']
            root = os.path.join(path, self.subdir)

        self.n_samples, self.seed = n_samples, seed
        self.n_components = n_components
        self.affine_align = affine_align

        self.affine_transform = AffineRegistration()
        self.model = PCA(n_components, seed=seed, whiten=True)
        self.sample_type = sample_type

        self.save_intermediate_steps = save_intermediate_steps

        super().__init__(root, transform, pre_transform)

        print(f"Loading dataset: {self}")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def subdir(self) -> str:
        return "paris_pca_skulls"

    @property
    def pca_model(self):
        return PCA.load(self.processed_dir, 'pca_model.joblib')

    # @property
    # def raw_file_names(self):
    #     file_list = glob.glob(self.raw_dir, '*/bone_template.ply')
    #     file_list.sort()
    #     return file_list

    @property
    def _raw_in_correspondence_data(self):
        ply_files = glob.glob(os.path.join(self.raw_dir, "*/bone_template.ply"))

        # format [(id, Data), ...]
        mesh_data = []
        for path in ply_files:
            data = read_ply(path)
            data['id'] = [int(path.split('/')[-2])]

            mesh_data.append(data)

        # sort files based on patient id
        mesh_data.sort(key = lambda d: d.id)

        return mesh_data

    @property
    def explained_variance_ratio(self):
        return self.pca_model.model.explained_variance_ratio_

    def _generate_synthetic_data(self, template, data):
        if self.sample_type == 'rand':
            np.random.seed(self.seed)
            z_samples = torch.randn(self.n_samples, self.n_components)

        if self.sample_type == 'linspace':
            z_samples = torch.linspace(-3, 3, self.n_samples)[:, None]
            z_samples = z_samples.repeat(1, self.n_components)

        generated_mesh_data = []
        for z in z_samples:
            t = template.clone()

            # convert to N×3 coordinates
            t.pos = self.model.inverse(z).view(-1, 3).float()
            t.z = z[None, :]

            generated_mesh_data.append(t)

        return generated_mesh_data

    @property
    def processed_file_names(self):
        return ['data.pt']

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
        if self.save_intermediate_steps:
            path = os.path.join(self.processed_dir, 'step_2_1')
            os.makedirs(path, exist_ok=True)
            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(path, self.processed_file_names[0]))

        # 3) get random template
        template = data_list[0].clone()

        # 4) affine align scans to template
        if self.affine_align:
            data_list = [self.affine_transform(template, data) for data in data_list]

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
        #     plotter.camera_position = 'xz'
        #     plotter.show()
        # exit()
        # PLOT RESULTS #

        # 4.1) store intermeditate results if self.save_intermediate_steps:
        if self.save_intermediate_steps:
            path = os.path.join(self.processed_dir, 'step_4_1')
            os.makedirs(path, exist_ok=True)
            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(path, self.processed_file_names[0]))

        # 5) fit the pca model
        data_list = torch.stack([d.pos for d in data_list if d])
        data_list = data_list.view(data_list.shape[0], -1)
        self.model.fit(data_list)
        self.model.save(self.processed_dir, 'pca_model.joblib')

        # 6) generate synthetic data using the pca model
        data_list = self._generate_synthetic_data(template, data_list)
        assert len(data_list) == self.n_samples

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
        #     plotter.camera_position = 'xz'
        #     plotter.show()
        # exit()
        # PLOT RESULTS #

        # 7) Store generated data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def train_dataloader(self, batch_size=32, shuffle=True, num_workers=8) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
