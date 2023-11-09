#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, "../")

import pyvista as pv
import gembed.models.point_flow_model as pfm
import gembed.models.point_manifold_flow_model as pmfm
import gembed.models.point_score_diffusion_model as psdm
import torch
import torch_geometric.transforms as tgt
from gembed.dataset import MSDLiver, MSDHippocampus


from synthesise import *
from register import *
from embed_shape_space import *
from interpolate import *
from atlas import *
from align import *

from density_map import *
from gembed import Configuration
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.dataset import (
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCasts,
    PittsburghDentalCastsCurvature,
)
from gembed.utils.dataset import train_valid_test_split
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from reconstruct import *
from gembed.core.optim import gradient_langevin_dynamics, gradient_ascent
import gembed.models.point_score_diffusion_model as psd
from transform import *
from datasets import *
from models import *
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(
    experiment_name,
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

        def reconstruct(model, dataset, device="cpu"):
            pl.seed_everything(42, workers=True)

            f_sample_points = lambda n_samples: (
                tgt.SamplePoints(n_samples)
                if hasattr(template, "face")
                else SubsetSample(n_samples)
            )

            # data transform
            T = tgt.Compose(
                [
                    f_sample_points(8192),
                    tgt.NormalizeScale(),
                ]
            )
            T_norm = tgt.Compose(
                [
                    tgt.NormalizeScale(),
                ]
            )

            model = model.to(device)

            for data in dataset:
                condition, params = model.inverse(
                    T(data.clone()).pos.to(device),
                    batch=None,
                    apply_stn=True,
                    return_params=True,
                )

                x = T_norm(data.clone()).pos.to(device)

                # apply stn
                if model.stn is not None:
                    x_aligned = model.stn.forward(x, None, params)
                else:
                    x_aligned = x

                # reconstruct the shape
                x1 = model.pdm_forward(
                    z=0.7*model.pdm.base_distribution.sample(int(8e2)).to(device),
                    condition=condition,
                    return_time_steps=False,
                )

                # only select points that are in [-1.1, 1.1]^3
                mask = x1.abs().max(1)[0] < 1.1
                x1 = x1[mask, :]

                # from gembed.core.optim import gradient_ascent
                # n_refinement_steps = 1000
                n_refinement_steps = 4
                if n_refinement_steps > 0:
                    x2 = gradient_ascent(
                        init_x=x1.requires_grad_(True),
                        f_grad=lambda x, b, c: model.pdm.score(
                            x, torch.Tensor([0.0]).to(x.device), b, c
                        ),
                        condition=condition,
                        batch_size=3000,  # 7000,
                        n_steps=n_refinement_steps,
                    ).detach()

                # invert stn
                if model.stn is not None:
                    x_rec = model.stn.inverse(x2, None, params)
                else:
                    x_rec = x2

                from gembed.vis import plot_objects

                plot_objects(
                    (T_norm(data).cpu(), None),
                    (x_aligned.cpu(), None),
                    (x1.cpu(), None),
                    (x2.cpu(), None),
                    (x_rec.cpu(), None),
                )
                # plot_objects(
                #     (x_aligned.cpu(), x_aligned.cpu()[:, 1]),
                #     (x_rec.cpu(), x_rec.cpu()[:, 1]),
                #     (x2.cpu(), x2.cpu()[:, 1]),
                # )

        # reconstruct(model, train[:4], device=device)
        # reconstruct(model, valid[:4], device=device)

        # 1) Evaluate data alignment
        # align_augmented_data(
        #     model,
        #     train,
        #     device=device,
        #     snapshot_root=snapshot_root,
        # )
        # align_augmented_data(
        #     model,
        #     valid,
        #     device=device,
        #     snapshot_root=snapshot_root,
        # )
        # align_augmented_data(
        #     model,
        #     test,
        #     device=device,
        #     snapshot_root=snapshot_root,
        # )

        # 2) Synthesise random shapes
        # sample_random_shape(
        #     model,
        #     train,
        #     n_random_shape_samples=10,
        #     n_random_point_samples=80000,
        #     n_refinement_steps=10,
        #     device=device,
        #     snapshot_root=snapshot_root,
        # )

        # 3) Density map
        #density_map(model, train[:4], device=device, plot_log_px=True)
        # density_map(model, valid[:4], device=device)
        # density_map(model, valid, device=device)
        # density_map(model, test, device=device)

        # 4) Qualitative Density
        # 5) Landmark Alignment
        # 6) Landmark Alignment (MSE)

        # 7) Register template to data
        def f_register_template(data, snapshot_root, dataset_name):
            if snapshot_root is not None:
                snapshot_root = os.path.join(snapshot_root, "register", dataset_name)

            register_template(
                model,
                data,
                template,
                device,
                snapshot_root=snapshot_root,
                plot_mesh_as_surface=False,
                n_template_samples=(int(8e5)),
                n_refinement_steps=10,
            )

        #f_register_template(train[1:], snapshot_root, "train")
        # f_register_template(valid[:4], snapshot_root, "valid")
        # f_register_template(test, snapshot_root, "test")

        # 8) Plot Shape Space
        Zs_train = embed_shape_space(
            model,
            train,
            device,
            # snapshot_root=os.path.join(snapshot_root, "shape_space"),
        )

        # 9) Interpolate Shapes
        # idx_1 = 0
        # idx_1 = 1
        # idx_2 = 5
        # idx_2 = 7
        # idx_2 = 9
        # # #idx_2 = 10
        # idx_2 = 26

        # dental curv
        # idx_1 = 2
        # idx_1 = 0

        # idx_2 = 6
        # idx_2 = 1
        #
        # hippocampus((0,1), (0, 4), (0,5), (0,7), (0,8))
        idx_1 = 0
        #idx_1 = 1
        #idx_2 = 1
        idx_2 = 7
        #idx_2 = 2

        # gauss
        # idx_1 = 2
        # idx_2 = 3

        # if snapshot_root is not None:
        #     snapshot_root = os.path.join(snapshot_root, "interpolate")
        # #record_distances(model, train, device=device)

        interpolate(
            model,
            train[idx_1],
            train[idx_2],
            n_refinement_steps=10,
            device=device,
            snapshot_root=snapshot_root,
            n_random_point_samples=int(8e4),
            Zs_train=Zs_train if "Zs_train" in locals() else None,
        )

        # 9) Atlas computation
        # idxs = [1, 21, 3]
        # idxs = [135, 112, 46] # <-
        # idxs = [207, 2, 97] # <-
        # idxs = [1, 207, 191]

        # import random
        # idxss = [[random.randint(0, len(train)-1), random.randint(0, len(train)-1), random.randint(0, len(train)-1)] for _ in range(100)]
        # if experiment_name == "hippocampus_mln":
        #     construct_template(
        #         model,
        #         train[idxs],
        #         init_template=template,
        #         local_metric_type="latent_mse",
        #         device=device,
        #     )
        # elif experiment_name == "hippocampus":
        # for idxs in idxss:
        #     print(idxs)
        # construct_template(
        #     model,
        #     train[idxs],
        #     init_template=template,
        #     local_metric_type="shape_mse",
        #     device=device,
        # )
        # construct_template(
        #     model,
        #     train[idxs],
        #     init_template=template,
        #     local_metric_type="latent_mse",
        #     device=device,
        # )
        # construct_template(
        #     model,
        #     train[idxs],
        #     init_template=template,
        #     local_metric_type="pullback_riemannian",
        #     device=device,
        # )
        # elif experiment_name == "skull":
        #     construct_template(
        #         model,
        #         train[:n_samples],
        #         init_template=template,
        #         local_metric_type="shape_mse",
        #         device=device,
        #     )
        # construct_template(
        #     model,
        #     train[:n_samples],
        #     init_template=template,
        #     local_metric_type="latent_mse",
        #     device=device,
        # )
        # construct_template(
        #     model,
        #     train[:n_samples],
        #     init_template=template,
        #     local_metric_type="pullback_riemannian",
        #     device=device,
        # )
        # elif experiment_name == "hippocampus_mln_hyper":
        #     construct_template(
        #         model,
        #         train[:n_samples],
        #         init_template=template,
        #         local_metric_type="latent_hyperbolic",
        #         device=device,
        #     )

        # 10) Classify shapes
        # if experiment_name == "dental" or experiment_name == "skull":
        #################################

        # sampled reconstruction (point GT)
        # sampled_reconstruction(
        #     model,
        #     train,
        #     device=device,
        #     sampled_vis_mesh=False,  # True,
        #     n_point_samples=80000,
        #     # n_refinement_steps=5,
        # )
        # sampled_reconstruction(
        #     model,
        #     valid,
        #     device=device,
        #     sampled_vis_mesh=True,
        #     n_input_samples=n_input_samples,
        # )

        # sampled reconstruction
        # sampled_reconstruction(model, train, device=device, n_input_samples=n_input_samples)
        # sampled_reconstruction(model, valid, device=device, n_input_samples=n_input_samples)

        # pc reconstruction
        # template_reconstruction(
        #     model, train, template, pc_template, device=device, sampled_vis_mesh=True, n_input_samples=n_input_samples
        # )
        # template_reconstruction(
        #     model, valid, template, pc_template, device=device, sampled_vis_mesh=True, n_input_samples=n_input_samples
        # )

        # pc reconstruction
        # template_reconstruction(model, train, template, pc_template, device=device, n_input_samples=n_input_samples)
        # template_reconstruction(model, valid, template, pc_template, device=device, n_input_samples=n_input_samples)

        # mesh reconstruction
        # template_reconstruction(
        #     model,
        #     train,
        #     template,
        #     mesh_template,
        #     device=device,
        #     n_input_samples=n_input_samples,
        # )
        # template_reconstruction(
        #     model,
        #     valid,
        #     template,
        #     mesh_template,
        #     device=device,
        #     n_input_samples=n_input_samples,
        # )

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
        # animated_reconstruction(model, valid, experiment_name, "valid", device=device, n_input_samples=n_input_samples)
        #
        # from animated_interpolation import animated_interpolation

        # animated_interpolation(
        #     model,
        #     Zs_train,
        #     experiment_name=experiment_name,
        #     split="train",
        #     device=device,
        #     start_pause_frames=0,
        #     shape_pause_frames=0,
        # )

        # Density map
        # density_map(model, train, device=device, n_input_samples=n_input_samples)
        # density_map(model, valid, device=device, n_input_samples=n_input_samples)

        # template generation
        # template = construct_template(model, train, template, n_samples=100, n_iters=5, device="cuda", n_input_samples=n_input_samples)
        # torch.save(template, "template.pt")
        # template = torch.load("template.pt")

        # corrected reconstruction
        # sampled_reconstruction_with_correction(
        #     model,
        #     train,
        #     device=device,
        #     sampled_vis_mesh=True,
        #     refinement_sampler=gradient_ascent,
        #     n_input_samples=n_input_samples,
        # )
        # sampled_reconstruction_with_correction(
        #     model, valid, device=device, sampled_vis_mesh=True, refinement_sampler=gradient_ascent
        # )
        # template_reconstruction_with_correction(
        #     model, train, template, mesh_template, device=device, refinement_sampler=gradient_ascent, n_input_samples=n_input_samples
        # )
        # template_reconstruction_with_correction(
        #     model, valid, template, mesh_template, device=device, refinement_sampler=gradient_ascent
        # )

        # template construction
        # frechet_mean(model, train, n_input_samples=n_input_samples, init_template=template, device=device)
        # construct_template(
        #     model,
        #     train,
        #     n_input_samples=n_input_samples,
        #     init_template=template,
        #     device=device,
        # )
