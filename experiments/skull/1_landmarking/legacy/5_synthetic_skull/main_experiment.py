#!/usr/bin/env python3
import sys

sys.path.insert(0, "../")

import os
import torch
import numpy as np
import gembed.models.linear_point_flow as lpf
from gembed import Configuration
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.utils.dataset import train_valid_test_split
from lightning.utilities.model_summary import ModelSummary
from gembed.dataset import LowResParisPCASkulls
import torch_geometric.transforms as tgt
from reconstruct import *

from transform import (
    ExcludeIDs,
    SubsetSample,
    SwapAxes,
    ThresholdImg2BinaryMask,
    BinaryMask2Surface,
    BinaryMask2Volume,
)
from gembed.transforms import RandomRotation, RandomTranslation, Clip


def load_config(experiment_name, template_samples=80000):
    print(f"Loading experiment: {experiment_name}")

    if experiment_name == "pca_surface":
        N_COMPONENTS = 28

        # INIT datasets
        dataset = LowResParisPCASkulls(
            pre_transform=tgt.NormalizeScale(),
            n_samples=100,
            affine_align=True,
            n_components=N_COMPONENTS,
        )

        model = lpf.SingleLaneLPF.load_from_checkpoint(
            "lightning_logs/pca_surface/point_flow/version_0/checkpoints/final_model.ckpt",
            n_components=N_COMPONENTS,
        ).eval()

    else:
        raise ValueError("Experiment name not valid")

    train, valid, test = train_valid_test_split(dataset)
    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    print(ModelSummary(model, -1))

    template = train[0].clone()
    mesh_template = train[0].clone()

    if hasattr(template, "face"):
        pc_template = tgt.SamplePoints(template_samples)(template.clone())
    else:
        pc_template = SubsetSample(template_samples)(template.clone())

    return model, template, mesh_template, pc_template, train, valid, test


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    experiment_name = sys.argv[-1]

    model, template, mesh_template, pc_template, train, valid, test = load_config(
        experiment_name
    )

    with torch.no_grad():
        # sampled reconstruction (point GT)
        # sampled_reconstruction(model, train, device=device, sampled_vis_mesh=False)
        # sampled_reconstruction(model, valid, device=device, sampled_vis_mesh=False)

        # sampled reconstruction
        # sampled_reconstruction(model, train, device=device)
        # sampled_reconstruction(model, valid, device=device)

        # pc reconstruction
        template_reconstruction(
            model, train, template, pc_template, device=device, sampled_vis_mesh=True
        )
        # template_reconstruction(
        #     model, valid, template, pc_template, device=device, sampled_vis_mesh=True
        # )

        # pc reconstruction
        # template_reconstruction(model, train, template, pc_template, device=device)
        # template_reconstruction(model, valid, template, pc_template, device=device)

        # mesh reconstruction
        # template_reconstruction(model, train, template, mesh_template, device=device)
        # template_reconstruction(model, valid, template, mesh_template, device=device)

        # Density map
        # density_map(model, train, device=device)
        density_map(model, valid, device=device)

        # breakpoint()
        # template generation
        # template = construct_template(model, train, template, n_samples=100, n_iters=5, device="cuda")
        # torch.save(template, "template.pt")
        # template = torch.load("template.pt")
