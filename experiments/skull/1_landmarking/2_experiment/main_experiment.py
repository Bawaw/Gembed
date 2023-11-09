#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, "../")

import gembed.models.point_flow_model as pfm
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
from transform import *
from glob import glob


def load_config(
    experiment_name,
    template_samples=80000,
    version=0,
    n_components=32,
    save_results=False,
):
    model_path = f"../1_train/lightning_logs/{experiment_name}/point_flow/version_{version}/checkpoints/final_model.ckpt"
    if not os.path.exists(model_path):
        model_path = glob(
            f"../1_train/lightning_logs/{experiment_name}/point_flow/version_{version}/checkpoints/epoch=*-step=*.ckpt"
        )[-1]

    print(f"Loading experiment: {experiment_name}, from path: {model_path}")

    if experiment_name == "skull":
        dataset = ParisVolumetricSkulls(
            pre_transform=tgt.Compose(
                [ThresholdImg2BinaryMask(), BinaryMask2Volume(), SwapAxes([2, 1, 0])]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            model_path, n_components=n_components, fourier_feature_scale=0.2
        ).eval()

    elif experiment_name == "hippocampus":
        dataset = MSDHippocampus(
            pre_transform=tgt.Compose(
                [
                    ThresholdImg2BinaryMask(threshold=0, components=None),
                    BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            model_path, n_components=n_components, fourier_feature_scale=0.4
        ).eval()

    elif experiment_name == "hippocampus_mln":
        dataset = MSDHippocampus(
            pre_transform=tgt.Compose(
                [
                    ThresholdImg2BinaryMask(threshold=0, components=None),
                    BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNMLNModel.load_from_checkpoint(
            model_path,
            n_components=n_components,
        ).eval()

    elif experiment_name == "hippocampus_mln_hyper":
        dataset = MSDHippocampus(
            pre_transform=tgt.Compose(
                [
                    # resample to lowest resolution
                    ThresholdImg2BinaryMask(threshold=0, components=None),
                    BinaryMask2Surface(reduction_factor=None),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNMLNModel.load_from_checkpoint(
            model_path, n_components=n_components, mln_metric="hyperbolic"
        ).eval()

    elif experiment_name == "brain":
        # INIT datasets
        dataset = ABCDBrain()
        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            model_path, n_components=n_components, fourier_feature_scale=0.8
        ).eval()

    # elif experiment_name == "dental":
    #     # INIT datasets
    #     dataset = PittsburghDentalCasts(
    #         pre_transform=tgt.Compose([SwapAxes([2, 0, 1]), InvertAxis(2)]),
    #     )

    #     model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
    #         model_path, n_components=n_components, fourier_feature_scale=0.4
    #     ).eval()

    elif experiment_name == "dental":
        # INIT datasets
        dataset = PittsburghDentalCastsCurvature(
            pre_transform=tgt.Compose(
                [
                    SwapAxes([2, 0, 1]),
                    InvertAxis(2),
                    SegmentMeshByCurvature(),
                    ClipMesh(),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            # model_path, n_components=n_components, fourier_feature_scale=0.4
            model_path,
            n_components=64,  # n_components,
            fourier_feature_scale=1.0,
        ).eval()

    elif experiment_name == "gauss":
        # INIT datasets
        from gembed.dataset.synthetic_double_gaussian_dataset import (
            SyntheticGaussianDataset,
        )

        dataset = SyntheticGaussianDataset(
            n_samples=100,
            n_point_samples=8192,
        )

        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            # model_path, n_components=n_components, fourier_feature_scale=0.4
            model_path,
            n_components=n_components,
            fourier_feature_scale=0.2,  # 1.0,
        ).eval()

    elif experiment_name == "gauss2":
        # INIT datasets
        from gembed.dataset.synthetic_double_gaussian_dataset import (
            SyntheticGaussianDataset2,
        )

        dataset = SyntheticGaussianDataset2(
            n_samples=100,
            n_point_samples=8192,
        )

        model = pfm.RegularisedPointFlowSTNModel.load_from_checkpoint(
            # model_path, n_components=n_components, fourier_feature_scale=0.4
            model_path,
            n_components=n_components,
            fourier_feature_scale=0.4,  # 1.0,
        ).eval()

    train, valid, test = train_valid_test_split(dataset)
    template = train[0].clone()

    # if skull we subsample the volumetric imagage
    # if experiment_name == "skull":
    #     template = SubsetSample(int(1e5))(template)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    model.pdm.layers.set_estimate_trace(False)
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
    print(device)

    experiment_name = sys.argv[-2]
    experiment_version = sys.argv[-1]
    (model, template, train, valid, test, snapshot_root,) = load_config(
        experiment_name,
        version=experiment_version,
        n_components=512,
        save_results=False,
    )

    train = train[:1]  # train[:4]

    with torch.no_grad():
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
        #     device=device,
        #     snapshot_root=snapshot_root,
        # )

        # 3) Density map
        # density_map(model, train, device=device)
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
                n_template_samples=(int(8e4)),
            )

        # f_register_template(train, snapshot_root, "train")
        # f_register_template(valid, snapshot_root, "valid")
        # f_register_template(test, snapshot_root, "test")

        # 8) Interpolate Shapes
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

        # gauss
        idx_1 = 2
        idx_2 = 3

        # if snapshot_root is not None:
        #     snapshot_root = os.path.join(snapshot_root, "interpolate")
        # #record_distances(model, train, device=device)

        # interpolate(
        #     model,
        #     train[idx_1],
        #     train[idx_2],
        #     local_metric_type="shape_mse",
        #     device=device,
        #     snapshot_root=snapshot_root,
        #     n_random_point_samples=int(8e4),
        # )
        # interpolate(
        #     model,
        #     train[idx_1],
        #     train[idx_2],
        #     local_metric_type="latent_mse",
        #     device=device,
        #     snapshot_root=snapshot_root,
        #     n_random_point_samples=int(8e4),
        # )
        # interpolate(
        #     model,
        #     train[idx_1],
        #     train[idx_2],
        #     local_metric_type="riemannian",
        #     device=device,
        #     snapshot_root=snapshot_root,
        #     riemannian_kwargs={
        #         "optim_n_random_point_samples": 10,
        #         "n_geodesic_cps": 6,
        #         "n_iters":10,
        #     },
        # )

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

        # 9) Plot Shape Space
        # embed_shape_space(
        #     model,
        #     train,
        #     device,
        #     umap_metric="euclidean",
        #     # snapshot_root=os.path.join(snapshot_root, "shape_space"),
        # )

        # 10) Classify shapes
        # if experiment_name == "dental" or experiment_name == "skull":
        #################################

        # sampled reconstruction (point GT)
        sampled_reconstruction(
            model,
            train,
            device=device,
            sampled_vis_mesh=False,  # True,
            n_point_samples=80000,
            # n_refinement_steps=5,
        )
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
        #     n_input_samples=n_input_samples,
        # )
        # animated_reconstruction(model, valid, experiment_name, "valid", device=device, n_input_samples=n_input_samples)

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
