#!/usr/bin/env python3
# import torch
# import numpy as np
# from torch_geometric.transforms import BaseTransform
# import functools
# from torch_geometric.data import Data
# import vtk
# from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
# import pyvista as pv
# from scipy.ndimage import gaussian_filter


# class ExtractSurfacePCByThreshold(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(
#         self,
#         threshold=500,
#         components=[1, 2, 3, 4, 5],
#         remove_pixel_information=True,
#         min_voxels=10000,
#         blur=True,
#     ):
#         self.threshold = threshold
#         self.components = components
#         self.min_voxels = min_voxels
#         self.blur = blur
#         self.remove_pixel_information = remove_pixel_information

#     def __call__(self, data):
#         img = data["vol"]

#         if self.blur:
#             img_filter = self.sitk.SmoothingRecursiveGaussianImageFilter()
#             img_filter.SetSigma(0.2)
#             img = img_filter.Execute(img)

#         # threshold
#         mask = img > self.threshold

#         if self.components is not None:
#             # CCA
#             component_image = self.sitk.ConnectedComponent(mask)

#             # sort the components based on size
#             sorted_component_image = self.sitk.RelabelComponent(
#                 component_image, sortByObjectSize=True
#             )

#             # select the components
#             mask = [
#                 sorted_component_image == c
#                 for c in self.components
#                 if self.sitk.GetArrayFromImage((sorted_component_image == c)).sum()
#                 > self.min_voxels
#             ]
#             mask = self.functools.reduce(lambda a, b: a | b, mask)

#         mask_arr = self.sitk.GetArrayFromImage(mask)
#         indices = np.stack(mask_arr.nonzero(), -1).astype(int)

#         new_data = Data(
#             pos=torch.FloatTensor(
#                 [img.TransformIndexToPhysicalPoint(i.tolist()) for i in indices]
#             )
#         )

#         # copy other data attributes
#         for k in data:
#             if self.remove_pixel_information and k == "vol":
#                 continue

#             new_data[k] = data[k]

#         return new_data


# class ThresholdImg2BinaryMask(BaseTransform):
#     import SimpleITK as sitk
#     import functools

#     def __init__(
#         self,
#         threshold=150,
#         components=[1, 2, 3, 4, 5],
#         remove_pixel_information=True,
#         min_voxels=10000,
#         blur=False,
#     ):
#         self.threshold = threshold
#         self.components = components
#         self.min_voxels = min_voxels
#         self.blur = blur
#         self.remove_pixel_information = remove_pixel_information

#     def __call__(self, data):
#         img = data["vol"]

#         if self.blur:
#             img_filter = self.sitk.SmoothingRecursiveGaussianImageFilter()
#             img_filter.SetSigma(1.0)
#             img = img_filter.Execute(img)

#         # threshold
#         mask = img > self.threshold

#         if self.components is not None:
#             # CCA
#             component_image = self.sitk.ConnectedComponent(mask)

#             # sort the components based on size
#             sorted_component_image = self.sitk.RelabelComponent(
#                 component_image, sortByObjectSize=True
#             )

#             # select the components
#             mask = [
#                 sorted_component_image == c
#                 for c in self.components
#                 if self.sitk.GetArrayFromImage((sorted_component_image == c)).sum()
#                 > self.min_voxels
#             ]
#             mask = self.functools.reduce(lambda a, b: a | b, mask)

#         data["vol"] = mask

#         # plot segmentation
#         # import pyvista as pv
#         # from gembed.vis import plot_sliced_volume

#         # segmentation_mask = self.sitk.GetArrayFromImage(mask)
#         # plot_sliced_volume(segmentation_mask)

#         return data


# class BinaryMask2Surface(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(
#         self,
#         remove_pixel_information=True,
#         n_smooth_iter=20,
#         reduction_factor=0.98,
#         pass_band=0.05,
#     ):
#         self.remove_pixel_information = remove_pixel_information
#         self.n_smooth_iter = n_smooth_iter
#         self.reduction_factor = reduction_factor
#         self.pass_band = pass_band

#     def __call__(self, data):
#         import pyvista as pv
#         from gembed.utils.adapter import vtk_to_torch_geometric_data

#         mask = data["vol"]

#         grid = pv.UniformGrid()
#         # note that sitk.mask.GetSize() != np.mask.shape
#         grid.dimensions = mask.GetSize()[::-1]
#         grid.origin = mask.GetOrigin()[::-1]
#         grid.spacing = mask.GetSpacing()[::-1]

#         grid["scalars"] = self.sitk.GetArrayFromImage(mask).flatten(order="F")

#         mesh = grid.contour(isosurfaces=2)
#         mesh = mesh.clean()

#         # mesh.plot(show_edges=True)
#         # pv.plot(mesh.points, render_points_as_spheres=True)

#         if self.n_smooth_iter > 0:
#             # mesh = mesh.smooth(n_iter=self.n_smooth_iter, relaxation_factor=0.1)
#             mesh = mesh.smooth_taubin(
#                 n_iter=self.n_smooth_iter, pass_band=self.pass_band
#             )

#         if self.reduction_factor is not None:
#             mesh = mesh.decimate(self.reduction_factor)
#         mesh = mesh.clean()

#         # mesh.plot()
#         # pv.plot(mesh.points, render_points_as_spheres=True)

#         new_data = vtk_to_torch_geometric_data(mesh)

#         # copy other data attributes
#         for k in data:
#             if self.remove_pixel_information and k == "vol":
#                 continue

#             new_data[k] = data[k]

#         return new_data


# class Resample(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(self, new_spacing):
#         self.new_spacing = new_spacing

#     def resample_volume(self, volume, new_spacing, interpolator=sitk.sitkLinear):
#         original_spacing = volume.GetSpacing()
#         original_size = volume.GetSize()
#         new_size = [
#             int(round(osz * ospc / nspc))
#             for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
#         ]

#         return self.sitk.Resample(
#             volume,
#             new_size,
#             self.sitk.Transform(),
#             interpolator,
#             volume.GetOrigin(),
#             new_spacing,
#             volume.GetDirection(),
#             0,
#             volume.GetPixelID(),
#         )

#     def __call__(self, data):
#         data["vol"] = self.resample_volume(data["vol"], self.new_spacing)
#         return data


# class SpacingFilter(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(self, spacing):
#         self.spacing = spacing

#     def __call__(self, data):
#         mask = data["vol"]

#         return (np.array(data["vol"].GetSpacing()) <= self.spacing).all()


# class BinaryMask2Surface(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(
#         self,
#         remove_pixel_information=True,
#         n_smooth_iter=20,
#         reduction_factor=0.98,
#     ):
#         self.remove_pixel_information = remove_pixel_information
#         self.n_smooth_iter = n_smooth_iter
#         self.reduction_factor = reduction_factor

#     def __call__(self, data):
#         import pyvista as pv
#         from gembed.utils.adapter import vtk_to_torch_geometric_data

#         mask = data["vol"]

#         grid = pv.UniformGrid()
#         grid.dimensions = mask.GetSize()[
#             ::-1
#         ]  # note that sitk.mask.GetSize() != np.mask.shape
#         grid.origin = mask.GetOrigin()
#         grid.spacing = mask.GetSpacing()
#         grid["scalars"] = self.sitk.GetArrayFromImage(mask).flatten(order="F")

#         mesh = grid.contour()

#         if self.n_smooth_iter > 0:
#             # mesh = mesh.smooth(n_iter=self.n_smooth_iter, relaxation_factor=0.1)
#             mesh = mesh.smooth_taubin(n_iter=self.n_smooth_iter, pass_band=0.05)

#         if self.reduction_factor is not None:
#             mesh = mesh.decimate(self.reduction_factor)

#         # mesh.plot()

#         new_data = vtk_to_torch_geometric_data(mesh)

#         # copy other data attributes
#         for k in data:
#             if self.remove_pixel_information and k == "vol":
#                 continue

#             new_data[k] = data[k]

#         return new_data


# class BinaryMask2Volume(BaseTransform):
#     import SimpleITK as sitk

#     def __init__(
#         self,
#         remove_pixel_information=True,
#     ):
#         self.remove_pixel_information = remove_pixel_information

#     def __call__(self, data):
#         mask = data["vol"]

#         mask_arr = self.sitk.GetArrayFromImage(mask)
#         indices = np.stack(mask_arr.nonzero(), -1).astype(int)

#         new_data = Data(
#             pos=torch.FloatTensor(
#                 [mask.TransformIndexToPhysicalPoint(i.tolist()) for i in indices]
#             )
#         )

#         # copy other data attributes
#         for k in data:
#             if self.remove_pixel_information and k == "vol":
#                 continue

#             new_data[k] = data[k]

#         return new_data


# class SubsetSample(BaseTransform):
#     def __init__(self, n_samples):
#         self.n_samples = n_samples

#     def __call__(self, data):
#         indices = np.random.choice(data.pos.shape[0], self.n_samples)

#         data.pos = data.pos[indices]

#         return data


# class SwapAxes(BaseTransform):
#     def __init__(
#         self,
#         axes=[0, 1, 2],
#     ):
#         self.axes = axes

#     def __call__(self, data):
#         data.pos = data.pos[:, self.axes]
#         return data


# class InvertAxis(BaseTransform):
#     def __init__(
#         self,
#         axis=1,
#     ):
#         self.axis = axis

#     def __call__(self, data):
#         data.pos[:, self.axis] = -data.pos[:, self.axis]
#         return data


# class ExcludeIDs(BaseTransform):
#     def __init__(self, ids):
#         self.ids = ids

#     def __call__(self, data):
#         return not (data["id"][0] in self.ids)


# class ThinPlateSplineAugmentation(BaseTransform):
#     def __init__(
#         self,
#         grid_size=6,
#         noise_sigma=0.03,
#         smooth_sigma=None,
#     ):
#         self.grid_size = grid_size
#         self.noise_sigma = noise_sigma
#         self.smooth_sigma = smooth_sigma

#     def __call__(self, data):
#         # grids
#         source_grid = np.stack(
#             np.meshgrid(
#                 np.linspace(-1.2, +1.2, self.grid_size),
#                 np.linspace(-1.2, +1.2, self.grid_size),
#                 np.linspace(-1.2, +1.2, self.grid_size),
#                 indexing="xy",
#             ),
#             -1,
#         )

#         # augmentation
#         target_grid = self.noise_sigma * np.random.randn(
#             self.grid_size, self.grid_size, self.grid_size, 3
#         )

#         if self.smooth_sigma is not None:
#             target_grid = gaussian_filter(target_grid, self.smooth_sigma)

#         target_grid = source_grid + target_grid

#         # grids to landmarks
#         source_landmarks = vtk.vtkPoints()
#         source_landmarks.SetData(numpy_to_vtk(source_grid.reshape(-1, 3)))
#         target_landmarks = vtk.vtkPoints()
#         target_landmarks.SetData(numpy_to_vtk(target_grid.reshape(-1, 3)))

#         # find deformtation from source to target grid
#         transform = vtk.vtkThinPlateSplineTransform()
#         transform.SetSourceLandmarks(source_landmarks)
#         transform.SetTargetLandmarks(target_landmarks)
#         transform.SetBasisToR()
#         transform.Update()

#         # transfer deformation to point cloud
#         warp = vtk.vtkTransformPolyDataFilter()
#         warp.SetTransform(transform)
#         warp.SetInputData(pv.PolyData(data.pos.numpy()))
#         warp.Update()

#         data.pos = torch.from_numpy(
#             vtk_to_numpy(warp.GetOutput().GetPoints().GetData())
#         )

#         return data


# class Clip(BaseTransform):
#     def __init__(self, threshold=1.2):
#         self.threshold = threshold

#     def __call__(self, data):
#         data.pos = data.pos[data.pos[:, 2] > self.threshold]
#         return data


# class RandomTranslation(BaseTransform):
#     def __init__(self, sigma=0.2):
#         self.sigma = sigma

#     def __call__(self, data):
#         b_transform = self.sigma * torch.randn(3)

#         data.pos = data.pos + b_transform

#         return data


# class RandomRotation(BaseTransform):
#     def __init__(self, sigma=0.5):
#         self.sigma = sigma

#     def __call__(self, data):
#         alpha, beta, gamma = self.sigma * torch.randn(3)

#         R = torch.stack(
#             [
#                 torch.stack(
#                     [
#                         torch.cos(beta) * torch.cos(gamma),
#                         torch.sin(alpha) * torch.sin(beta) * torch.cos(gamma)
#                         - torch.cos(alpha) * torch.sin(gamma),
#                         torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)
#                         + torch.sin(alpha) * torch.sin(gamma),
#                     ],
#                     -1,
#                 ),
#                 torch.stack(
#                     [
#                         torch.cos(beta) * torch.sin(gamma),
#                         torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma)
#                         + torch.cos(alpha) * torch.cos(gamma),
#                         torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)
#                         - torch.sin(alpha) * torch.cos(gamma),
#                     ],
#                     -1,
#                 ),
#                 torch.stack(
#                     [
#                         -torch.sin(beta),
#                         torch.sin(alpha) * torch.cos(beta),
#                         torch.cos(alpha) * torch.cos(beta),
#                     ],
#                     -1,
#                 ),
#             ],
#             -1,
#         )

#         # [N x 3] x [3 x 3] = [N x 3]
#         data.pos = data.pos @ R

#         return data


# class RandomJitter(object):
#     def __init__(self, scale, dim=3, distribution="normal"):
#         if distribution == "normal":
#             self.distribution = torch.distributions.Normal(
#                 torch.tensor([0.0] * dim), torch.tensor([scale] * dim)
#             )
#         elif distribution == "laplace":
#             self.distribution = torch.distributions.Laplace(
#                 torch.tensor([0.0] * dim), torch.tensor([scale] * dim)
#             )
#         else:
#             raise ValueError("Only normal/laplace distributions supported for Jitter.")

#     def __call__(self, data):
#         data.pos = data.pos + self.distribution.sample(data.pos.shape[:1])
#         return data

#     def __repr__(self):
#         return "{}({})".format(self.__class__.__name__, self.distribution)


# from scipy.signal import argrelmin
# from scipy.stats import gaussian_kde

# import torch
# from torch.nn import Linear, Parameter
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
# import torch_geometric.transforms as tgt


# class SegmentMeshByCurvature(object):
#     def __init__(
#         self,
#         smooth=True,
#         smooth_hops=2,
#         smooth_steps=5,
#     ):
#         self.smooth = smooth
#         self.smooth_hops = smooth_hops
#         self.smooth_steps = smooth_steps

#     class MajorityVoteSmoothing(MessagePassing):
#         def __init__(self):
#             super().__init__(aggr="mean")

#         def forward(self, x, edge_index):
#             # add x_i to the neighbourhood
#             edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#             # majority vote on binary values, if >50% of nodes == 1 -> 1
#             out = 1.0 * (0.5 < self.propagate(edge_index, x=x))

#             return out

#         def message(self, x_j):
#             return x_j

#     def __call__(self, data):
#         # fit KDE
#         kernel = gaussian_kde(data.x[:, 0])

#         # compute density for entire interval
#         x_axis = torch.linspace(-6e5, data.x.max(), 1000)[:, None]
#         densities = kernel(x_axis[:, 0])

#         # select the local minima
#         minima = argrelmin(densities)[0]

#         # if no local minima take default threshold value
#         if len(minima) == 0:
#             threshold = -5e5
#         else:
#             # select the minima that has the lowest density
#             threshold = x_axis[minima[densities[minima].argmin()]]

#         # create binary mask
#         data.x = 1.0 * (data.x < threshold)

#         ####### PLOT KDE #######
#         # import matplotlib.pyplot as plt

#         # fig, ax = plt.subplots()
#         # ax.plot(x_axis[:, 0], densities)
#         # ax.axvline(x=threshold, color="red")
#         # plt.show()
#         ########################

#         # smooth the result
#         if self.smooth:
#             data = tgt.FaceToEdge(remove_faces=False)(data)
#             for i in range(self.smooth_hops):
#                 data = tgt.TwoHop()(data)

#             smooth = SegmentMeshByCurvature.MajorityVoteSmoothing()

#             for _ in range(self.smooth_steps):
#                 x = smooth(data.x, data.edge_index)

#                 # if no changes stop smoothing
#                 if (x == data.x).all():
#                     break
#                 else:
#                     data.x = x

#             del data.edge_index

#         ###### START PLOT ######
#         # from gembed.vis import Plotter

#         # pl = Plotter()
#         # pl.add_generic(data, scalars=data.x)
#         # pl.show()
#         #######################

#         return data


# import pyvista as pv
# from gembed.utils import adapter


# class ClipMesh(object):
#     def __init__(self, threshold=0, fill_holes=True, fill_hole_area=1, remove_x=True):
#         self.threshold = threshold
#         self.fill_holes = fill_holes
#         self.fill_hole_area = fill_hole_area
#         self.remove_x = remove_x

#     def __call__(self, data):
#         pv_data = adapter.torch_geomtric_data_to_vtk(data.pos, data.face)

#         # clip mesh
#         pv_data["segmentation"] = data.x
#         pv_data = pv_data.clip_scalar("segmentation", invert=False)

#         # fill potential holes
#         if self.fill_holes:
#             pv_data = pv_data.fill_holes(self.fill_hole_area)

#         # plot result
#         # pv_data.plot()

#         # assign new vertices and faces to original mesh
#         data.face = torch.from_numpy(pv_data.faces).reshape(-1, 4).permute(1, 0)[1:]
#         data.pos = torch.from_numpy(pv_data.points)

#         # remove the segmentation mask
#         if self.remove_x:
#             del data.x

#         return data
