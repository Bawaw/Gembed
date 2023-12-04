import numpy as np
from torch_geometric.transforms import BaseTransform
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

class ThinPlateSplineAugmentation(BaseTransform):
    def __init__(
        self,
        grid_size=6,
        noise_sigma=0.03,
        smooth_sigma=None,
    ):
        self.grid_size = grid_size
        self.noise_sigma = noise_sigma
        self.smooth_sigma = smooth_sigma

    def __call__(self, data):
        # grids
        source_grid = np.stack(
            np.meshgrid(
                np.linspace(-1.2, +1.2, self.grid_size),
                np.linspace(-1.2, +1.2, self.grid_size),
                np.linspace(-1.2, +1.2, self.grid_size),
                indexing="xy",
            ),
            -1,
        )

        # augmentation
        target_grid = self.noise_sigma * np.random.randn(
            self.grid_size, self.grid_size, self.grid_size, 3
        )

        if self.smooth_sigma is not None:
            target_grid = gaussian_filter(target_grid, self.smooth_sigma)

        target_grid = source_grid + target_grid

        # grids to landmarks
        source_landmarks = vtk.vtkPoints()
        source_landmarks.SetData(numpy_to_vtk(source_grid.reshape(-1, 3)))
        target_landmarks = vtk.vtkPoints()
        target_landmarks.SetData(numpy_to_vtk(target_grid.reshape(-1, 3)))

        # find deformtation from source to target grid
        transform = vtk.vtkThinPlateSplineTransform()
        transform.SetSourceLandmarks(source_landmarks)
        transform.SetTargetLandmarks(target_landmarks)
        transform.SetBasisToR()
        transform.Update()

        # transfer deformation to point cloud
        warp = vtk.vtkTransformPolyDataFilter()
        warp.SetTransform(transform)
        warp.SetInputData(pv.PolyData(data.pos.numpy()))
        warp.Update()

        data.pos = torch.from_numpy(
            vtk_to_numpy(warp.GetOutput().GetPoints().GetData())
        )

        return data
