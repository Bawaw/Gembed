#!/usr/bin/env python3
import os
import glob

from .abstract_dataset import AbstractDataset
from gembed import Configuration

class MSDLiver(AbstractDataset):
    """ Segmented livers from the medical segmentation decathlon dataset."""

    @property
    def _raw_data(self):
        import SimpleITK as sitk

        dcm_files = glob.glob(os.path.join(self.raw_dir, "liver_*.nii.gz"))

        # format [(id, Data), ...]
        raw_data = []
        for path in dcm_files:
            data = {"vol": sitk.ReadImage(path)}
            data["id"] = [int(os.path.basename(path).split("_")[1].split(".")[0])]

            raw_data.append(data)

        # sort files based on patient id
        raw_data.sort(key=lambda d: d["id"])

        return raw_data

    def process(self):
        return super().process()
