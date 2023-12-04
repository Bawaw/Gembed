from scipy.signal import argrelmin
from scipy.stats import gaussian_kde

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.transforms as tgt
from torch_geometric.transforms import BaseTransform


class SegmentMeshByCurvature(BaseTransform):
    def __init__(
        self,
        smooth=True,
        smooth_hops=2,
        smooth_steps=5,
    ):
        self.smooth = smooth
        self.smooth_hops = smooth_hops
        self.smooth_steps = smooth_steps

    class MajorityVoteSmoothing(MessagePassing):
        def __init__(self):
            super().__init__(aggr="mean")

        def forward(self, x, edge_index):
            # add x_i to the neighbourhood
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            # majority vote on binary values, if >50% of nodes == 1 -> 1
            out = 1.0 * (0.5 < self.propagate(edge_index, x=x))

            return out

        def message(self, x_j):
            return x_j

    def __call__(self, data):
        # fit KDE
        kernel = gaussian_kde(data.x[:, 0])

        # compute density for entire interval
        x_axis = torch.linspace(-6e5, data.x.max(), 1000)[:, None]
        densities = kernel(x_axis[:, 0])

        # select the local minima
        minima = argrelmin(densities)[0]

        # if no local minima take default threshold value
        if len(minima) == 0:
            threshold = -5e5
        else:
            # select the minima that has the lowest density
            threshold = x_axis[minima[densities[minima].argmin()]]

        # create binary mask
        data.x = 1.0 * (data.x < threshold)

        ####### PLOT KDE #######
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # ax.plot(x_axis[:, 0], densities)
        # ax.axvline(x=threshold, color="red")
        # plt.show()
        ########################

        # smooth the result
        if self.smooth:
            data = tgt.FaceToEdge(remove_faces=False)(data)
            for i in range(self.smooth_hops):
                data = tgt.TwoHop()(data)

            smooth = SegmentMeshByCurvature.MajorityVoteSmoothing()

            for _ in range(self.smooth_steps):
                x = smooth(data.x, data.edge_index)

                # if no changes stop smoothing
                if (x == data.x).all():
                    break
                else:
                    data.x = x

            del data.edge_index

        ###### START PLOT ######
        # from gembed.vis import Plotter

        # pl = Plotter()
        # pl.add_generic(data, scalars=data.x)
        # pl.show()
        #######################

        return data
