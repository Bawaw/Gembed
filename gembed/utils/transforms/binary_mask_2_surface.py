from torch_geometric.transforms import BaseTransform

class BinaryMask2Surface(BaseTransform):
    import SimpleITK as sitk

    def __init__(
        self,
        remove_pixel_information=True,
        n_smooth_iter=20,
        reduction_factor=0.98,
        pass_band=0.05,
    ):
        self.remove_pixel_information = remove_pixel_information
        self.n_smooth_iter = n_smooth_iter
        self.reduction_factor = reduction_factor
        self.pass_band = pass_band

    def __call__(self, data):
        import pyvista as pv
        from gembed.utils.adapter import vtk_to_torch_geometric_data

        mask = data["vol"]

        grid = pv.ImageData()
        # note that sitk.mask.GetSize() != np.mask.shape
        grid.dimensions = mask.GetSize()[::-1]
        grid.origin = mask.GetOrigin()[::-1]
        grid.spacing = mask.GetSpacing()[::-1]

        grid["scalars"] = self.sitk.GetArrayFromImage(mask).flatten(order="F")

        mesh = grid.contour(isosurfaces=2)
        mesh = mesh.clean()

        # mesh.plot(show_edges=True)
        # pv.plot(mesh.points, render_points_as_spheres=True)

        if self.n_smooth_iter > 0:
            # mesh = mesh.smooth(n_iter=self.n_smooth_iter, relaxation_factor=0.1)
            mesh = mesh.smooth_taubin(
                n_iter=self.n_smooth_iter, pass_band=self.pass_band
            )

        if self.reduction_factor is not None:
            mesh = mesh.decimate(self.reduction_factor)
        mesh = mesh.clean()

        # mesh.plot()
        # pv.plot(mesh.points, render_points_as_spheres=True)

        new_data = vtk_to_torch_geometric_data(mesh)

        # copy other data attributes
        for k in data:
            if self.remove_pixel_information and k == "vol":
                continue

            new_data[k] = data[k]

        return new_data
