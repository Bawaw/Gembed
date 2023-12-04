#!/usr/bin/env python3

import os
import torch
import pyvista as pv
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch_geometric.transforms as tgt

from gembed.dataset import load_dataset
from gembed.core.optim import gradient_ascent
from gembed.models import PointScoreDiffusionModel
from gembed.utils.dataset import train_valid_test_split
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary

PYVISTA_PLOT_KWARGS = {
    "color" : "#cccccc",
}

PYVISTA_SAVE_KWARGS = {
    "window_size" : [4000, 4000],
    "color" : "#cccccc",
    "point_size" : 20,
    "cmap" : "cool",
}

def pathcat(str_1, str_2):
    """ Concatenates two strings representing file or directory paths.

        Args:
          str_1 (str): The first part of the path.
          str_2 (str): The second part of the path.

        Returns:
            str or None: The concatenated path. Returns `None` if either `str_1` or `str_2` is `None`.
    """

    if str_1 is None or str_2 is None:
        return None

    return os.path.join(str_1, str_2)

def clean_result(X):
    """ Remove points that are outside of cube [1.1]^3.

        Args:
            X (torch.Tensor): The input data to be cleaned.
        Returns:
            torch.Tensor: The cleaned input data.

    """

    mask = X.abs().max(1)[0] < 1.1
    X = X[mask, :]

    return X

def refine_result(model, X, Z, n_refinement_steps, batch_size=3000):
    """ Refines the input data `X` using gradient ascent based on a given model and condition Z.

        Args:
            model (YourModel): The model used for refining the input.
            X (torsnapshot_rootch.Tensor): The input data to be refined.
            Z (torch.Tensor): The condition for the refinement.
            n_refinement_steps (int): The number of refinement steps to perform using gradient ascent.
            batch_size (int, optional): The batch size used for gradient ascent. Default is 3000.

        Returns:
            torch.Tensor: The refined input data after the specified number of refinement steps.
    """

    X_refined = X.clone()
    if n_refinement_steps > 0:
        X_refined = gradient_ascent(
            init_x = X_refined.requires_grad_(True),
            f_grad=lambda x, b, c: model.pdm.score(
                x, torch.Tensor([0.0]).to(x.device), b, c
            ),
            condition=Z.clone(),
            batch_size=batch_size,
            n_steps=n_refinement_steps,
        ).detach()

    return X_refined


def load_experiment(args):
    pl.seed_everything(42, workers=True)

    # READ COMMAND ARGS
    assert len(args) >=2, f(
        "Expected at least 2 command arguments. \\"
        "Usage: experiment_name.py experiment_name model_version [-d] [-s] \\\\"
        "Options: \\  -d dark theme\\  -s save results in output/"
    )

    experiment_name = args[0]
    version = args[1]
    save_results = True if "-s" in args else False
    theme = "dark" if "-d" in args else "light"

    # SETUP EXPERIMENT
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if theme == "dark":
        pv.set_plot_theme("dark")
        sns.set(style="ticks", context="talk")
        plt.style.use("dark_background")
    elif theme == "light":
        pv.set_plot_theme("document")

    if save_results:
        snapshot_root = os.path.join("output", experiment_name, f"version_{version}")
        os.makedirs(snapshot_root, exist_ok=True)
    else:
        snapshot_root = None

    # LOAD DATA & MODEL
    dataset = load_dataset(experiment_name, train=False)
    model = PointScoreDiffusionModel.load(experiment_name, version=version).to(device)

    train, valid, test = train_valid_test_split(dataset)
    template = train[0].clone()

    # EXPERIMENT DEPENDENT UTILITY FUNCTIONS
    T_sample = tgt.SamplePoints(8192) if hasattr(template, "face") else SubsetSample(8192)
    f_refine = lambda X, Z, n_refinement_steps: refine_result(model, X, Z, n_refinement_steps, batch_size=3000)

    # SUMMARISE EXPERIMENT
    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    print(ModelSummary(model, -1))

    assert model.training == False, "Model still in training mode."

    return (
        model,
        T_sample,
        f_refine,
        template,
        train,
        valid,
        test,
        device,
        snapshot_root,
    )

