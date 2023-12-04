#!/usr/bin/env python3
import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from helper import load_experiment, pathcat, PYVISTA_PLOT_KWARGS, PYVISTA_SAVE_KWARGS

def plot_result(X, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if file_path is None:
        plot_objects((X.cpu(), None), snapshot_file_name=file_path, **PYVISTA_PLOT_KWARGS)
    else:
        plot_objects((X.cpu(), None), snapshot_file_name=file_path, **PYVISTA_SAVE_KWARGS)

def main(
    model,
    T_sample,
    f_refine,
    n_random_shape_samples=5,
    n_random_point_samples=int(8e4),
    n_refinement_steps=0,
    device="cpu",
    file_path=None,
):
    pl.seed_everything(42, workers=True)

    # get seeds for shapes
    print("TODO: use base distribution as seed after refactoring...")
    #Zs = 0.8*model.ltn.base_distribution.sample(n_random_shape_samples).to(device)
    Zs = 0.8*torch.randn(n_random_shape_samples, 512).to(device)

    # get seeds for point configuration
    z = 0.8*model.pdm.base_distribution.sample(n_random_point_samples).to(device)

    for i, Z in enumerate(Zs.unsqueeze(1)):
        # synthesise new shape
        H, X = model.forward(Z=Z, z=z, apply_ltn=True, apply_pdm=True)

        # refine the new shape
        X = f_refine(X, H, n_refinement_steps)

        # plot the results
        plot_result(X, pathcat(file_path, f"X_{i}.png"))



if __name__ == "__main__":
    import sys
    #assert False, "This function still has to be tested!!!"

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        main(model, T_sample, f_refine, device=device, file_path=file_path)
