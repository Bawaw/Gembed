#!/usr/bin/env python3

import os
import torch
import pytorch_lightning as pl
import torch_geometric.transforms as tgt

from gembed.vis import plot_objects
from helper import load_experiment, pathcat, PYVISTA_SAVE_KWARGS, PYVISTA_PLOT_KWARGS

def plot_result(template, target, registered_template, refined_registered_template, scalars, file_path, plot_mesh_as_surface=False):
    # remove mesh information
    if hasattr(registered_template, "face") and not plot_mesh_as_surface:
        del template.face
        del target.face
        del registered_template.face
        del refined_registered_template.face

    if file_path is None:
        kwargs = PYVISTA_PLOT_KWARGS
        kwargs["cmap"] = "prism"

        plot_objects(
            (template.cpu(), scalars),
            (target.cpu(), None),
            (registered_template.cpu(), scalars),
            (refined_registered_template.cpu(), scalars),
            snapshot_file_name=file_path, **kwargs)

    else:
        kwargs = PYVISTA_SAVE_KWARGS
        kwargs["cmap"] = "prism"

        os.makedirs(file_path, exist_ok=True)
        plot_objects((template.cpu(), scalars), snapshot_file_name=pathcat(file_path, "template.png"), **kwargs)
        plot_objects((target.cpu(), None), snapshot_file_name=pathcat(file_path, "target.png"), **kwargs)
        plot_objects((registered_template.cpu(), scalars), snapshot_file_name=pathcat(file_path, "registered_template.png"), **kwargs)
        plot_objects((refined_registered_template.cpu(), scalars), snapshot_file_name=pathcat(file_path, "refined_registered_template.png"), **kwargs)


def main(
    model,
    T_sample,
    f_refine,
    dataset,
    template,
    device="cpu",
    plot_mesh_as_surface=False,
    n_refinement_steps=6,
    n_template_samples=int(8e4),
    file_path=None
):
    pl.seed_everything(42, workers=True)

    assert not(plot_mesh_as_surface and n_template_samples is not None),(
        "Mesh reconstruction not supported.")

    T_norm = tgt.Compose(
        [
            tgt.NormalizeScale(),
        ]
    )

    # setup template
    template = T_norm(template.clone())

    if n_template_samples is not None:
        T_template_sampler = (
            tgt.SamplePoints(n_template_samples) if hasattr(template, "face")
            else SubsetSample(n_template_samples)
        )

        template_processed = T_template_sampler(
            template.clone()
        ).to(device) # draw new samples from template
    else:
        template_processed = template.clone().to(device)

    scalars_template = template_processed.pos[:, 1].cpu()

    z_template = model.pdm_inverse(
        x=template_processed.pos,
        condition=model.inverse(
            T_sample(template.clone()).pos.to(device),
            batch=None,
            apply_stn=True,
        ),
        apply_stn=True,
    )


    for i, data in enumerate(dataset):
        X_data = T_norm(data.clone()).to(device)

        # register template to X_data and gain: X_reg ≈ X_data
        X_reg = template.clone()
        Z_reg, stn_params = model.inverse(
            X=T_sample(X_data.clone()).pos,
            batch=None,
            apply_stn=True,
            return_params=True,
        )
        X_reg.pos = model.pdm_forward(
            z=z_template,
            condition=Z_reg,
            return_time_steps=False,
        )

        # refine
        X_reg_refined = template.clone()
        X_reg_refined.pos = f_refine(X_reg.pos, Z_reg, n_refinement_steps)

        # invert the spatial alignment
        if model.stn is not None:
            X_reg.pos = model.stn.inverse(X_reg.pos, None, stn_params)
            X_reg_refined.pos = model.stn.inverse(X_reg_refined.pos, None, stn_params)

        # plot the result
        plot_result(template_processed, X_data, X_reg, X_reg_refined, scalars_template,
                    pathcat(file_path, str(data.id)), plot_mesh_as_surface)


if __name__ == "__main__":
    import sys

    (
        model, T_sample, f_refine, template, train, valid, test, device, file_path
     ) = load_experiment(sys.argv[1:])

    file_path = pathcat(file_path, str(os.path.basename(__file__)).split(".")[0])

    with torch.no_grad():
        main(model, T_sample, f_refine, train[:5], template, device=device, file_path=pathcat(file_path, "train"))
        main(model, T_sample, f_refine, valid[:5], template, device=device, file_path=pathcat(file_path, "valid"))
        main(model, T_sample, f_refine, test[:5], template, device=device, file_path=pathcat(file_path, "test"))
