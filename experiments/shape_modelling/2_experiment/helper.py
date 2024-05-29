#!/usr/bin/env python3

import json
import os

import matplotlib.pyplot as plt
import lightning as pl
import pyvista as pv
import seaborn as sns
import torch
import torch_geometric.transforms as tgt
from lightning.pytorch.utilities.model_summary.model_summary import \
    ModelSummary

from gembed import Configuration
from gembed.core.optim import gradient_ascent
from gembed.dataset import load_dataset
from gembed.models import PointScoreDiffusionModel
from gembed.utils.dataset import train_valid_test_split
from gembed.utils.transforms.subset_sample import SubsetSample

def get_registration_cdim(experiment_name):
    if "brain" in experiment_name:
        cdim = 1
    else:
        cdim = 0

    return cdim

def get_plot_cdim(experiment_name):
    if "skull" in experiment_name:
        cdim = 1
    else:
        cdim = 0

    return cdim

def pyvista_plot_kwargs(experiment_name):
    PYVISTA_PLOT_KWARGS = {
        "color" : "#cccccc",
        "point_size" : 10,
        "cmap" : "cool",
    }
    # PYVISTA_PLOT_KWARGS = {
    #     "color" : "#cccccc",
    #     "cmap" : "cool",
    #     #"clim" : [-1, 1],
    # }

    if "brain" in experiment_name:
        PYVISTA_PLOT_KWARGS["camera_position"] = [(8, 0, 0), (0, 0, 0), (0, 0, 1)]
    elif "dental" in experiment_name:
        PYVISTA_PLOT_KWARGS["camera_position"] = [(-6, -6, 6), (0, 0, 0), (0, 0, 1)]
    elif "skull" in experiment_name:
        PYVISTA_PLOT_KWARGS["camera_position"] = [(-6, -6, 0), (0, 0, 0), (0, 0, 1)]
    else:
        PYVISTA_PLOT_KWARGS["camera_position"] = [(-6, -6, 0), (0, 0, 0), (0, 0, 1)]

    return PYVISTA_PLOT_KWARGS

def pyvista_save_kwargs(experiment_name):
    PYVISTA_SAVE_KWARGS = {
        #"window_size" : [4000, 4000],
        "color" : "#cccccc",
        "point_size" : 10,
        "cmap" : "cool",
    }
    # PYVISTA_SAVE_KWARGS = {
    #     "window_size" : [4000, 4000],
    #     "color" : "#cccccc",
    #     "point_size" : 20,
    #     "cmap" : "cool",
    # }

    if "brain" in experiment_name:
        PYVISTA_SAVE_KWARGS["camera_position"] = [(8, 0, 0), (0, 0, 0), (0, 0, 1)]
    elif "dental" in experiment_name:
        PYVISTA_SAVE_KWARGS["camera_position"] = [(-6, -6, 6), (0, 0, 0), (0, 0, 1)]
    elif "skull" in experiment_name:
        PYVISTA_SAVE_KWARGS["camera_position"] = [(-6, -6, 0), (0, 0, 0), (0, 0, 1)]
    else:
        PYVISTA_SAVE_KWARGS["camera_position"] = [(-6, -6, 0), (0, 0, 0), (0, 0, 1)]

    return PYVISTA_SAVE_KWARGS

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

def refine_result(model, X, Z, n_refinement_steps=20, batch_size=30000, score_threshold=None):
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
    f_grad=lambda x, b, c: model.pdm.score(
        x, torch.Tensor([0.0]).to(x.device), b, c
    )

    if n_refinement_steps > 0:
        X_refined = gradient_ascent(
            init_x = X_refined.requires_grad_(True),
            f_grad=f_grad,
            condition=Z.clone(),
            batch_size=batch_size,
            n_steps=n_refinement_steps,
            step_size=1e-5,
            #step_size=1e-1,
        ).detach()

    if score_threshold is not None:
        #score = f_grad(X_refined, torch.zeros(X_refined.shape[0]).long(), Z).pow(2).sum(-1)
        score = f_grad(X_refined, torch.zeros(X_refined.shape[0]).long(), Z).abs().mean(-1)
        X_refined = X_refined[score < score_threshold]

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

    path_model_kwargs = os.path.join(Configuration()["Paths"]["MODEL_CONFIG_DIR"], experiment_name + ".json")
    
    model = PointScoreDiffusionModel.load(
        **json.load(open(path_model_kwargs)),
        version=version
    ).to(device)

    train, valid, test = train_valid_test_split(dataset)
    template = train[0].clone()

    # EXPERIMENT DEPENDENT UTILITY FUNCTIONS
    if experiment_name == "brain" or experiment_name =="dental": # use the same amount of samples as used in training
        n_point_samples = 2**15
    else: 
        n_point_samples = 8192

    T_sample = tgt.SamplePoints(n_point_samples) if hasattr(template, "face") else SubsetSample(n_point_samples)
    f_refine = lambda X, Z, n_refinement_steps: refine_result(model, X, Z, n_refinement_steps, batch_size=3000)

    # SUMMARISE EXPERIMENT
    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    print(ModelSummary(model, -1))

    assert model.training == False, "Model still in training mode."

    return (
        experiment_name,
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

