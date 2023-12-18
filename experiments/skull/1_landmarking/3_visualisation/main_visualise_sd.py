#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, "../")

import pyvista as pv
import torch
from gembed.dataset import MSDHippocampus

from visualise_space import *

from gembed.dataset import (
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCasts,
    PittsburghDentalCastsCurvature,
)
from gembed.utils.dataset import train_valid_test_split
from lightning.utilities.model_summary.model_summary import ModelSummary
from transform import *
from datasets import *
from models import *
from glob import glob
from interpolate_animated import interpolate_animated
from reconstruct_animated import reconstruct_animated
from vis_alignment import vis_alignment
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(
    experiment_name,
    template_samples=80000,
    version=0,
    n_components=32,
    save_results=False,
):
    dataset = load_dataset(experiment_name, train=False)
    model = load_point_score_diffusion_model(experiment_name, version=version)

    train, valid, test = train_valid_test_split(dataset)
    template = train[0].clone()

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    print(ModelSummary(model, -1))

    if save_results:
        snapshot_root = os.path.join("output", experiment_name, f"version_{version}")
        os.makedirs(snapshot_root, exist_ok=True)
    else:
        snapshot_root = None

    return (
        model,
        template,
        train,
        valid,
        test,
        snapshot_root,
    )


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    theme = "dark"
    if theme == "dark":
        pv.set_plot_theme("dark")
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
    elif theme == "light":
        pv.set_plot_theme("document")

    experiment_name = sys.argv[-2]
    experiment_version = sys.argv[-1]
    (model, template, train, valid, test, snapshot_root,) = load_config(
        experiment_name,
        version=experiment_version,
        n_components=512,
        save_results=False,
    )

    # train = train[:1]
    # train = train[:4]

    with torch.no_grad():
        # visualise dataset pre and post alignment
        vis_alignment(model, train[:5], device)

        # embed data in spaces
        # vis_spaces(model, train, 0, 7, device=device)

        # animate the reconstruction
        # animated_reconstruction(
        #     model,
        #     train,
        #     experiment_name,
        #     "train",
        #     device=device,
        #     start_pause_frames=10,
        #     pre_refinement_pause_frames=25,
        #     end_pause_frames=50,
        # )
        # animated_reconstruction(
        #     model,
        #     valid,
        #     experiment_name,
        #     "valid",
        #     device=device,
        #     n_input_samples=n_input_samples,
        # )

        # animated interpolation
        # Zs_train = embed_shape_space(model, train[:5], device)
        # animated_interpolation(
        #     model,
        #     Zs_train,
        #     experiment_name=experiment_name,
        #     split="train",
        #     device=device,
        #     start_pause_frames=0,
        #     shape_pause_frames=0,
        # )
