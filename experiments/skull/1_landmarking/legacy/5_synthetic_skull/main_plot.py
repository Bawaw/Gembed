#!/usr/bin/env python3

from gembed.vis import Plotter
from gembed.vis import plot_objects
from gembed.dataset import LowResParisPCASkulls
import torch_geometric.transforms as tgt

if __name__ == "__main__":

    dataset = LowResParisPCASkulls(
        pre_transform=tgt.NormalizeScale(),
        #transform=Compose([tgt.SamplePoints(2048)]),
        n_samples=100,
        affine_align=True,
        n_components=28,
    )

    # plot meshes
    for mesh in dataset:
        plot_objects((mesh, None))

    # plot meshes on top of each other
    plt = Plotter()
    for surf in vol_dataset:
        surf = tgt.Center()(vol)
        plt.add_generic(vol)

    plt.camera_position = [(0, -5, 0), (0, 0, 0), (0, 0, 1)]
    plt.show()
