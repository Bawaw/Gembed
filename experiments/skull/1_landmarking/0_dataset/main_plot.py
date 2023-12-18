#!/usr/bin/env python3
import sys

sys.path.insert(0, "../")

import os

import lightning as pl
from gembed.dataset import (
    MSDLiver,
    ParisVolumetricSkulls,
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCasts,
    PittsburghDentalCastsCurvature,
)
from gembed.utils.dataset import train_valid_test_split
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as tgt
from gembed.vis import plot_objects
from gembed import Configuration

from transform import *


def replot_same_data_object_for_n_iter(dataset, index=0, n_iter=100):
    for i in range(n_iter):
        d = dataset[index]
        plot_objects((d, None), color="#eeeeee")


def iteratively_plot_dataset(dataset, plot_surface=True):
    for i, d in enumerate(dataset):
        if not plot_surface and hasattr(d, "face"):
            del d.face

        print(f"Data index: {i}, object id: {d.id}")
        plot_objects((d, None), color="#eeeeee", show_grid=True)


def iteratively_plot_dataset_on_top(dataset):
    print(f"Surface id: {d1.id}, Volume id: {d2.id}")
    plt = Plotter()

    for d in dataset:
        plt.add_generic(d, color="#eeeeee")

    plt.show()


def iteratively_plot_datasets_side_by_side(dataset_1, dataset_2):
    print(f"Dataset 1 size: {len(dataset_1)}, dataset 2 size: {len(dataset_2)}")
    print(f"Dataset 1 scale: {dataset_1.scale}, Vol dataset size: {dataset_2.scale}")

    for d1, d2 in zip(dataset_1, dataset_2):
        plot_objects((d1, None), (d2, None))


def plot_datasets_on_top(dataset_1, dataset_2):
    for d1, d2 in zip(dataset_1, dataset_2):
        print(f"Surface id: {d1.id}, Volume id: {d2.id}")

        plt = Plotter()
        plt.add_generic(d1, color="red")
        plt.add_generic(d2, color="green")
        plt.show()


if __name__ == "__main__":

    pl.seed_everything(42, workers=True)

    # # INIT datasets

    # LIVER DATASET
    # liver_dataset = MSDLiver(
    #     pre_filter = SpacingFilter([1, 1, 1]),
    #     pre_transform=tgt.Compose(
    #         [
    #             # resample to lowest resolution
    #             Resample([1, 1, 1]),
    #             ThresholdImg2BinaryMask(threshold=0),
    #             BinaryMask2Surface(n_smooth_iter=50, pass_band=0.000001, reduction_factor=0.90),
    #         ]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #             #tgt.Center()
    #             tgt.NormalizeScale(),
    #     #         tgt.SamplePoints(8192),
    #             # ThinPlateSplineAugmentation(),
    #             # tgt.RandomShear(0.1),
    #             # tgt.NormalizeScale(),
    #             # RandomRotation(sigma=0.2),
    #             # RandomTranslation(sigma=0.1),
    #         ]
    #     ),
    # )
    # replot_same_data_object_for_n_iter(liver_dataset)
    # iteratively_plot_dataset(liver_dataset)
    #
    # abcd brain dataset
    # brain_dataset = ABCDBrain(
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #         ]
    #     ),
    # )
    # # replot_same_data_object_for_n_iter(brain_dataset)
    # iteratively_plot_dataset(brain_dataset, plot_surface=True)

    # abcd brain dataset
    # def segment_dentition(data):
    # from torch_geometric.utils import k_hop_subgraph
    # from sklearn.decomposition import PCA

    # def local_flatness(data, node_idx, n_hops=50):
    #     # 1) for every node find the neighbourhood of n-hops
    #     subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    #         node_idx, n_hops, data.edge_index
    #     )

    #     # 2) For every neighbourhood select the positions
    #     neigh_pos = data.pos[subset]

    #     # 3) Project the positions onto 2D plane and then reconstruct the patch
    #     pca = PCA(n_components=2)
    #     neigh_pos_rec = pca.inverse_transform(pca.fit_transform(neigh_pos))

    #     # 4) compute the error from the reconstruction
    #     # error = (neigh_pos - neigh_pos_rec).pow(2).sum(-1).mean()
    #     error = (neigh_pos - neigh_pos_rec).abs().sum(-1).mean()

    #     return error

    # # unfortunately we have to iterate this because the neighbourhood size is not fixed and thus we can not batch it
    # # discrete_curvature = torch.tensor([local_flatness(data, node_idx) for node_idx in range(data.pos.shape[0])])

    # from joblib import Parallel, delayed

    # import time

    # start = time.time()
    # discrete_curvature = torch.Tensor(
    #     Parallel(prefer="threads", n_jobs=8)(
    #         delayed(local_flatness)(data, node_idx)
    #         for node_idx in range(data.pos.shape[0])
    #     )
    # )
    # end = time.time()
    # print(end - start)

    # from gembed.utils import adapter
    # import vtk
    # from sklearn.metrics import pairwise as dist
    # from copy import copy

    # pv_data = adapter.torch_geomtric_data_to_vtk(data.pos, data.face)

    # # # pv_data = pv_data.decimate_pro(0.5)#.smooth(100)

    # # # https://vtk.org/doc/nightly/html/classvtkPCACurvatureEstimation.html#details
    # cc = vtk.vtkPCACurvatureEstimation()
    # # cc = vtk.vtkPCANormalEstimation()
    # cc.SetInputData(pv_data)
    # cc.SetSampleSize(10)
    # # cc.SetSearchModeToKNN()
    # cc.Update()

    # output = cc.GetOutput()
    # pca_curvature_directions = vtk.util.numpy_support.vtk_to_numpy(
    #     output.GetPointData().GetArray(0)
    # )
    # discrete_curvature = pca_curvature_directions[:, 0]

    # # curvature_normals = pv_data.compute_normals(cell_normals=False, auto_orient_normals=True)["Normals"]

    # # #d = dist.euclidean_distances
    # # d = dist.cosine_similarity
    # # scalars = pca_curvature_directions[:, 2] #d(pca_curvature_directions, np.array([[0, 0, 1]]))
    # # scalars2 = d(curvature_normals, np.array([[0, 0, 1]]))
    # # #scalars3 = dist.paired_cosine_distances(curvature_normals, pca_curvature_directions)
    # # scalars3 = dist.paired_euclidean_distances(curvature_normals, pca_curvature_directions)

    # # scalars = dist.euclidean_distances(pca_curvature_directions, np.array([[0, 0, 1]]))
    # # scalars = np.max(pca_curvature_directions, 1)

    # # pca_curvature_directions

    # pl = pv.Plotter()
    # # pl = pv.Plotter(shape=(1, 4))

    # pl.subplot(0, 0)
    # pl.add_mesh(copy(pv_data), scalars=discrete_curvature)
    # # pl.add_arrows(pv_data.points, pca_curvature_directions, mag=0.1)

    # # pl.subplot(0, 1)
    # # pl.add_mesh(copy(pv_data), scalars = scalars2)
    # # pl.add_arrows(pv_data.points, curvature_normals, mag=0.1)

    # # pl.subplot(0, 2)
    # # pl.add_mesh(copy(pv_data), scalars = scalars3)
    # # pl.add_arrows(pv_data.points, curvature_normals, mag=0.1)

    # # pl.link_views()
    # pl.show()

    # # pv_data = pv_data.smooth(1000).curvature("mean")

    # # pv_data.plot(scalars=curvature)

    # dental_cast_dataset = PittsburghDentalCasts(
    #     pre_transform=tgt.Compose([SwapAxes([2, 0, 1]), InvertAxis(2)]),
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #             tgt.FaceToEdge(remove_faces=False),
    #             segment_dentition,
    #             # tgt.SamplePoints(8192),
    #             RandomJitter(0.01),
    #         ]
    #     ),
    # )
    # replot_same_data_object_for_n_iter(dental_cast_dataset)
    # iteratively_plot_dataset(dental_cast_dataset, plot_surface=True)

    # dental cast curvature

    dental_cast_dataset = PittsburghDentalCastsCurvature(
        pre_transform=tgt.Compose([
            SwapAxes([2, 0, 1]),
            InvertAxis(2),
            SegmentMeshByCurvature(),
            ClipMesh(),
        ]),
        transform=tgt.Compose(
            [
                tgt.NormalizeScale(),
                #tgt.SamplePoints(8192),
                ThinPlateSplineAugmentation(),
                tgt.RandomShear(0.05),
                tgt.NormalizeScale(),
                # RandomRotation(sigma=0.2),
                # RandomTranslation(sigma=0.1),
            ]
        ),
    )
    dental_cast_dataset, valid, test = train_valid_test_split(dental_cast_dataset)
    # replot_same_data_object_for_n_iter(dental_cast_dataset)
    iteratively_plot_dataset(dental_cast_dataset, plot_surface=True)
    # hippocampus dataset
    # hippocampus_dataset = MSDHippocampus(
    #     pre_transform=tgt.Compose(
    #         [
    #             # resample to lowest resolution
    #             ThresholdImg2BinaryMask(threshold=0, components=None),
    #             BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
    #         ]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #             tgt.SamplePoints(8192),
    #             ThinPlateSplineAugmentation(noise_sigma=0.1),
    #             tgt.RandomShear(0.1),
    #             tgt.NormalizeScale(),
    #             RandomRotation(sigma=0.2),
    #             RandomTranslation(sigma=0.1),
    #         ]
    #     ),
    # )
    # replot_same_data_object_for_n_iter(hippocampus_dataset)
    # iteratively_plot_dataset(hippocampus_dataset, plot_surface=True)

    # Clean Mesh DATASET
    # paris_mesh_dataset = CleanParisMeshSkulls(
    #     # transform=tgt.Compose(
    #     #     [
    #     #         # tgt.SamplePoints(8192),
    #     #         # tgt.RandomFlip(axis=0),
    #     #         # ThinPlateSplineAugmentation(),
    #     #         # tgt.RandomScale([0.8, 1.0]),
    #     #         # tgt.RandomShear(0.05),
    #     #     ]
    #     # ),
    # )

    # iteratively_plot_dataset(paris_mesh_dataset)

    # skull_dataset = ParisVolumetricSkulls(
    #     pre_transform=tgt.Compose(
    #         [ThresholdImg2BinaryMask(), BinaryMask2Volume(), SwapAxes([2, 1, 0])]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #             tgt.NormalizeScale(),
    #             SubsetSample(8192),
    #             tgt.RandomFlip(axis=0),
    #             ThinPlateSplineAugmentation(),
    #             tgt.RandomScale([0.8, 1.0]),
    #             tgt.RandomShear(0.05),
    #             tgt.NormalizeScale(),
    #             # RandomRotation(sigma=0.2),
    #             # RandomTranslation(sigma=0.1),
    #         ]
    #     ),
    # )

    # iteratively_plot_dataset(skull_dataset, plot_surface=False)

    # surf_dataset = CleanParisMeshSkulls(
    #     pre_transform=Center(),
    # )
    # vol_dataset = ParisVolumetricSkulls(
    #     pre_filter=ExcludeIDs([15]),
    #     pre_transform=Compose(
    #         [ExtractSurfacePCByThreshold(), SwapAxes([2, 1, 0]), Center()]
    #     ),
    # )

    # surf_dataset = ParisVolumetricSkulls(
    #     root=os.path.join(Configuration()["Paths"]["DATA_DIR"], "paris_volumetric_skulls_isosurface"),
    #     pre_filter=ExcludeIDs([15]),
    #     pre_transform=tgt.Compose(
    #         [
    #             ThresholdImg2BinaryMask(),
    #             BinaryMask2Surface(),
    #             SwapAxes([2, 1, 0])
    #          ]
    #     ),
    # )
    # vol_dataset = ParisVolumetricSkulls(
    #     pre_filter=ExcludeIDs([15]),
    #     pre_transform=tgt.Compose(
    #         [
    #             ThresholdImg2BinaryMask(),
    #             BinaryMask2Volume(),
    #             SwapAxes([2, 1, 0]),
    #         ]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #           Clip()
    #           # tgt.Center(),
    #           # SubsetSample(4096),
    #           # tgt.RandomFlip(axis=0),
    #           # ThinPlateSplineAugmentation(),
    #           # tgt.RandomScale([0.8, 1.]),
    #           # tgt.RandomShear(0.05),
    #         ]
    #     ),
    # )
    # vol_dataset = ParisVolumetricSkulls(
    #     root=os.path.join(
    #         Configuration()["Paths"]["DATA_DIR"], "paris_volumetric_skulls_isosurface"
    #     ),
    #     pre_filter=ExcludeIDs([15]),
    #     pre_transform=tgt.Compose(
    #         [
    #             ThresholdImg2BinaryMask(),
    #             BinaryMask2Surface(),
    #             SwapAxes([2, 1, 0]),
    #         ]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #             tgt.Center(),
    #             tgt.RandomFlip(axis=0),
    #             ThinPlateSplineAugmentation(),
    #             tgt.RandomScale([0.8, 1.]),
    #             tgt.RandomShear(0.05),
    #         ]
    #     ),
    # )

    # surf_dataset = ParisVolumetricSkulls(
    #     root=os.path.join(
    #         Configuration()["Paths"]["DATA_DIR"], "paris_volumetric_skulls_isosurface"
    #     ),
    #     pre_filter=ExcludeIDs([15]),
    #     pre_transform=tgt.Compose(
    #         [
    #             ThresholdImg2BinaryMask(),
    #             BinaryMask2Surface(),
    #             SwapAxes([2, 1, 0]),
    #         ]
    #     ),
    #     transform=tgt.Compose(
    #         [
    #             tgt.Center(),
    #         ]
    #     ),
    # )
