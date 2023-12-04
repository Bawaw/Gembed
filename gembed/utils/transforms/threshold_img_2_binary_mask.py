
from torch_geometric.transforms import BaseTransform

class ThresholdImg2BinaryMask(BaseTransform):
    import SimpleITK as sitk
    import functools

    def __init__(
        self,
        threshold=150,
        components=[1, 2, 3, 4, 5],
        remove_pixel_information=True,
        min_voxels=10000,
        blur=False,
    ):
        self.threshold = threshold
        self.components = components
        self.min_voxels = min_voxels
        self.blur = blur
        self.remove_pixel_information = remove_pixel_information

    def __call__(self, data):
        img = data["vol"]

        if self.blur:
            img_filter = self.sitk.SmoothingRecursiveGaussianImageFilter()
            img_filter.SetSigma(1.0)
            img = img_filter.Execute(img)

        # threshold
        mask = img > self.threshold

        if self.components is not None:
            # CCA
            component_image = self.sitk.ConnectedComponent(mask)

            # sort the components based on size
            sorted_component_image = self.sitk.RelabelComponent(
                component_image, sortByObjectSize=True
            )

            # select the components
            mask = [
                sorted_component_image == c
                for c in self.components
                if self.sitk.GetArrayFromImage((sorted_component_image == c)).sum()
                > self.min_voxels
            ]
            mask = self.functools.reduce(lambda a, b: a | b, mask)

        data["vol"] = mask

        # plot segmentation
        # import pyvista as pv
        # from gembed.vis import plot_sliced_volume

        # segmentation_mask = self.sitk.GetArrayFromImage(mask)
        # plot_sliced_volume(segmentation_mask)

        return data
