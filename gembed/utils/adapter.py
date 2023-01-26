#!/usr/bin/env python3

import torch
import pyvista as pv
from torch_geometric.data import Data

def torch_geomtric_data_to_vtk(verts, faces):
    if faces.shape[0] != 3:
        faces = torch.t(faces)

    face_size = torch.ones(1, faces.shape[1])*3
    faces = torch.cat((face_size.long(), faces))
    faces = faces.permute(1, 0).reshape(-1,).numpy()
    return pv.PolyData(verts.numpy(), faces)

def vtk_to_torch_geometric_data(data):
    faces = torch.from_numpy(data.faces)
    faces = faces.reshape(-1, 4).permute(1, 0)[1:]
    vertices = torch.from_numpy(data.points)
    return Data(pos=vertices, face=faces)
