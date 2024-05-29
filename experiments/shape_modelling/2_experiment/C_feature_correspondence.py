import os
import torch
import lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis.plotter import Plotter
from helper import load_experiment, pathcat, pyvista_plot_kwargs, pyvista_save_kwargs

def run(model, template, dataset, device, file_path, ref_index = 18000):
    pl.seed_everything(42, workers=True)
    T_norm = tgt.NormalizeScale()

    #TODO: remove this <<
    print("WARNING: using highest z-value for reference index")
    ref_index = template.pos[:, -1].argmax(dim=0)
    # >> remove

    # 1) template to point feature representation
    template = T_norm(template.clone())
    z_template = model.sdm.feature_nn.feature_forward(template.pos.to(device), None, False).cpu()

    # 2) feature representation of reference point
    template_color = torch.zeros(z_template.shape[0])
    template_color[ref_index] = 1
    z_ref_template = z_template[ref_index]
    pos_ref_template = template.pos[ref_index]

    for data in dataset:
        data = T_norm(data.clone())

        # 3) target to point feature representation
        z = model.sdm.feature_nn.feature_forward(data.pos.to(device), None, False).cpu()

        # 4) distance between point features and reference feature
        a, c = 100, 1
        distance_color = a*torch.exp(-(z - z_ref_template[None]).pow(2)/2*c)

        # 5) plot the result
        plotter = Plotter(shape = (1, 2))

        plotter.subplot(0, 0)
        plotter.add_generic(template.cpu(), cmap="cool", color = "#cccccc") 
        plotter.add_generic(pos_ref_template[None], color="red", point_size=20, render_points_as_spheres=True) 

        plotter.subplot(0, 1)
        plotter.add_generic(data.cpu(), scalars=distance_color, cmap="cool", color = "#cccccc") 

        plotter.link_views()
        plotter.camera_position = [(-5, -5, 0), (0, 0, 0), (0, 0, 1)]

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
        run(model, template, train[:5], device=device, file_path=pathcat(file_path, "train"))
        run(model, template, valid[:5], device=device, file_path=pathcat(file_path, "valid"))
        run(model, template, test[:5], device=device, file_path=pathcat(file_path, "test"))

if __name__ == "__main__":
    main()