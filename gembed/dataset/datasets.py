#!/usr/bin/env python3

from gembed.utils.transforms import ThresholdImg2BinaryMask, BinaryMask2Surface, SubsetSample, SwapAxes, InvertAxis, SegmentMeshByCurvature, SegmentMesh, ThresholdImg2BinaryMask, BinaryMask2Volume
import torch_geometric.transforms as tgt
from gembed.dataset import (
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCastsCurvature,
    ParisVolumetricSkulls,
)


def load_dataset(experiment_name, train=True):
    if experiment_name == "hippocampus":
        if train:
            transform = tgt.Compose(
                [
                    tgt.SamplePoints(8192),
                    tgt.NormalizeScale(),
                    # ThinPlateSplineAugmentation(noise_sigma=0.1),
                    # tgt.RandomShear(0.1),
                    # tgt.NormalizeScale(),
                    # RandomRotation(sigma=0.2),
                    # RandomTranslation(sigma=0.1),
                ]
            )

        else:
            transform = None

        dataset = MSDHippocampus(
            pre_transform=tgt.Compose(
                [
                    ThresholdImg2BinaryMask(threshold=0, components=None),
                    BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
                ]
            ),
            transform=transform,
        )

    elif experiment_name == "skull":
        if train:
            transform = tgt.Compose(
                [
                    SubsetSample(8192),
                    tgt.NormalizeScale(),
                    # tgt.RandomFlip(axis=0),
                    # tgt.RandomScale([0.8, 1.0]),
                    # ThinPlateSplineAugmentation(),
                    # tgt.RandomShear(0.05),
                    # RandomRotation(sigma=0.2),
                    # RandomTranslation(sigma=0.1),
                ]
            )

        else:
            transform = None

        dataset = ParisVolumetricSkulls(
            pre_transform=tgt.Compose(
                [ThresholdImg2BinaryMask(), BinaryMask2Volume(), SwapAxes([2, 1, 0])]
            ),
            transform=transform,
        )[:100]

    elif experiment_name == "brain":
        if train:
            transform = tgt.Compose(
                [
                    tgt.SamplePoints(8192),
                    tgt.NormalizeScale(),
                    # RandomRotation(sigma=0.2),
                    # RandomTranslation(sigma=0.1),
                ]
            )

        else:
            transform = None

        dataset = ABCDBrain(transform=transform)[:100]

    elif experiment_name == "dental":
        if train:
            transform = tgt.Compose(
                [
                    tgt.SamplePoints(8192),
                    tgt.NormalizeScale(),
                    # ThinPlateSplineAugmentation(),
                    # tgt.RandomShear(0.05),
                    # tgt.NormalizeScale(),
                    # RandomRotation(sigma=0.2),
                    # RandomTranslation(sigma=0.1),
                ]
            )

        else:
            transform = None

        dataset = PittsburghDentalCastsCurvature(
            pre_transform=tgt.Compose(
                [
                    SwapAxes([2, 0, 1]),
                    InvertAxis(2),
                    SegmentMeshByCurvature(),
                    SegmentMesh(),
                ]
            ),
            transform=transform,
        )[:100]
    else:
        raise ValueError(f"Invalid experiment name: {experiment_name}")

    print(f"Loaded the dataset: {dataset}, with transform: {dataset.transform}, pre-transform: : {dataset.pre_transform}")

    return dataset
