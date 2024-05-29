import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.utils.transforms import RandomRotation, RandomTranslation
from gembed.vis.plotter import Plotter
from gembed.vis import plot_objects
from helper import load_experiment, pathcat, pyvista_save_kwargs, pyvista_plot_kwargs


def run(model, T_sample, dataset, device, file_path):
    pl.seed_everything(42, workers=True)

    T_norm = tgt.NormalizeScale()
    T_augment = tgt.Compose(
        [RandomRotation(sigma=0.2), RandomTranslation(sigma=0.1)]
    )

    plotter = Plotter(shape=(1, 4))

    for i, data in enumerate(dataset):
        # 1) original data
        X = T_sample(T_norm(data.clone()))
    
        plotter.subplot(0, 0)
        plotter.add_generic(X.pos, scalars=i*torch.ones(X.pos.shape[0]))

        # 2) superimpose shape
        X_aligned = model.stn.forward(X.clone().pos.to(device), None).cpu()

        plotter.subplot(0, 1)
        plotter.add_generic(X_aligned, scalars=i*torch.ones(X.pos.shape[0]))

        # 3) augmented 
        X_augmented = T_augment(X.clone())

        plotter.subplot(0, 2)
        plotter.add_generic(X_augmented.pos, scalars=i*torch.ones(X.pos.shape[0]))

        # 4) augmented and superimposed
        X_aligned_2 = model.stn.forward(X_augmented.clone().pos.to(device), None).cpu()

        plotter.subplot(0, 3)
        plotter.add_generic(X_aligned_2, scalars=i*torch.ones(X.pos.shape[0]))

    plotter.show()

def main():
    import sys

    (
        experiment_name, model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    global EXPERIMENT_NAME 
    EXPERIMENT_NAME = experiment_name

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        run(model, T_sample, train[:5], device=device, file_path=pathcat(file_path, "train"))
        run(model, T_sample, valid[:5], device=device, file_path=pathcat(file_path, "valid"))
        run(model, T_sample, test[:5], device=device, file_path=pathcat(file_path, "test"))

if __name__ == "__main__":
    main()
