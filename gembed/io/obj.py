import torch
import numpy as np

from torch_geometric.data import Data

def read_obj(path, mesh_color = False):
    import pywavefront
    scene = pywavefront.Wavefront(path, collect_faces = True)
    vert = np.array(scene.vertices)
    col = vert[:,3:]
    vert = vert[:,:3]
    faces = np.array(scene.mesh_list[0].faces).T

    template = Data()
    template.pos = torch.tensor(vert).float()
    template.face = torch.tensor(faces).long()
    if mesh_color:
        template.mesh_color = torch.tensor(col)
    return template

def save_obj(mesh, path, mesh_color=False):
    pos = mesh.pos
    if mesh_color:
        # add color as vertex columns
        pos = torch.cat([pos, mesh.mesh_color], 1)

    # add vertex prefix
    pos = np.concatenate([np.array(['v']*pos.shape[0])[:,None], pos.numpy()], 1)

    # add face prefix, obj is 1 indexed
    face = mesh.face.T + 1
    face = np.concatenate([np.array(['f']*face.shape[0])[:,None], face], 1)

    obj_file = open(path, "w")
    for row in pos:
        np.savetxt(obj_file, row[None], fmt='%s')
    np.savetxt(obj_file, np.array([]), fmt='%s')
    for row in face:
        np.savetxt(obj_file, row[None], fmt='%s')

    obj_file.close()
