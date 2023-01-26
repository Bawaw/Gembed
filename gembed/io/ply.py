#!/usr/bin/env python3

import numpy
import torch
import numpy as np
from torch_geometric.data import Data


def read_ply(path, read_color=False, read_features=False):
    from plyfile import PlyData

    ply_data = PlyData.read(path)

    data_template = Data()

    try:
        data_template.pos = torch.from_numpy(np.stack(
            [ply_data['vertex']['x'], ply_data['vertex']['y'],ply_data['vertex']['z']],-1))

        data_template.face = torch.from_numpy(np.stack(
            ply_data['face']['vertex_indices']).T).long()
    except:
        print("Failed to read {}, returning None.".format(path))
        return None


    if read_features:
        data_template.x = torch.from_numpy(ply_data['vertex']['quality'])[:,None]

    if read_color:
        data_template.mesh_color = torch.from_numpy(np.stack(
            [ply_data['vertex']['red'], ply_data['vertex']['green'],
            ply_data['vertex']['blue']], -1))

    return data_template
