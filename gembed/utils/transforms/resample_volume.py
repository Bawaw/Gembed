from torch_geometric.transforms import BaseTransform

class ResampleVolume(BaseTransform):
    import SimpleITK as sitk

    def __init__(self, new_spacing):
        self.new_spacing = new_spacing

    def resample_volume(self, volume, new_spacing, interpolator=sitk.sitkLinear):
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        return self.sitk.Resample(
            volume,
            new_size,
            self.sitk.Transform(),
            interpolator,
            volume.GetOrigin(),
            new_spacing,
            volume.GetDirection(),
            0,
            volume.GetPixelID(),
        )

    def __call__(self, data):
        data["vol"] = self.resample_volume(data["vol"], self.new_spacing)
        return data
