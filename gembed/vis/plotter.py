#!/usr/bin/env python3

import torch
import numpy as np
import pyvista as pv
from torch_geometric.data import Data
from gembed.utils import adapter


class Plotter(pv.Plotter):
    def add_mesh(self, mesh, **kwargs):
        # assert (not isinstance(mesh, Data)
        #         or ('batch' not in mesh.keys
        #             or mesh.batch.max().item() == 0)), """
        #   plotter does not accept batched tensors
        # """

        assert not isinstance(mesh, Data) or (
            "cpu" in mesh.pos.device.type and "cpu" in mesh.face.device.type
        ), """
          plotter does not accept meshes on GPU
        """

        if isinstance(mesh, Data):
            mesh = adapter.torch_geomtric_data_to_vtk(mesh.pos, mesh.face)

        super().add_mesh(mesh, **kwargs)
        return mesh

    def add_pointcloud(self, pc, **kwargs):
        # assert not isinstance(pc, Data) or (
        #     "batch" not in pc.keys or pc.batch.max() == 0
        # ), """
        #   plotter does not accept batched tensors
        # """

        assert not isinstance(pc, Data) or (
            "cpu" in pc.pos.device.type
        ), """
          plotter does not accept meshes on GPU
        """

        if isinstance(pc, Data):
            pc = pc.pos
        if isinstance(pc, torch.Tensor):
            pc = pc.numpy()

        if pc.shape[1] < 3:
            pc = np.concatenate(
                [pc, np.zeros([pc.shape[0], 3 - pc.shape[1]], dtype=pc.dtype)], axis=1
            )
        point_cloud = pv.PolyData(pc)

        super().add_mesh(point_cloud, **kwargs)
        return point_cloud

    def add_volume(self, vol, **kwargs):
        assert not isinstance(vol, Data) or (
            "batch" not in vol.keys or vol.batch.max() == 0
        ), """
          plotter does not accept batched tensors
        """

        assert (
            not isinstance(vol, Data) or "cpu" in vol.vol.device.type
        ), """
          plotter does not accept voles on GPU
        """

        if isinstance(vol, Data):
            vol = vol.vol[0].numpy()

        super().add_volume(vol, **kwargs)
        return vol

    def update_coordinates(self, x, **kwargs):
        if isinstance(x, Data):
            x = x.pos
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        if x.shape[1] < 3:
            x = np.concatenate(
                [x, np.zeros([x.shape[0], 3 - x.shape[1]], dtype=x.dtype)], axis=1
            )

        super().update_coordinates(x, **kwargs)

    def add_generic(self, x, **kwargs):
        # if torch geometric object
        if isinstance(x, Data):
            if hasattr(x, "vol") and x["vol"] is not None:
                return self.add_volume(x, **kwargs)
            if hasattr(x, "face") and x.face is not None:
                return self.add_mesh(x, **kwargs)
            else:
                return self.add_pointcloud(x, **kwargs)

        # if pyvista polydata object
        if isinstance(x, pv.PolyData):
            super().add_mesh(x, **kwargs)

        # if tensor or numpy array (mesh not supported)
        else:
            if isinstance(x, torch.Tensor):
                if x.requires_grad:
                    x = x.detach()
                x = x.numpy()
            if len(x.shape) > 2:
                return self.add_volume(x, **kwargs)
            else:
                return self.add_pointcloud(x, **kwargs)
