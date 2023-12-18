#!/usr/bin/env python3


import sys

sys.path.insert(0, "../")

import os

import gembed.models.point_flow_model as pfm
from gembed import Configuration
import torch_geometric.transforms as tgt
import lightning as pl
from gembed.dataset import (
    MSDLiver,
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCasts,
    PittsburghDentalCastsCurvature,
)
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.utils.dataset import train_valid_test_split
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from gembed.transforms import RandomRotation, RandomTranslation, Clip
from transform import *
from math import isclose

import numpy as np

def load_config(experiment_name):
    print(f"Loading experiment: {experiment_name}")

    if experiment_name == "skull":

        dataset = ParisVolumetricSkulls(
            pre_transform=tgt.Compose(
                [ThresholdImg2BinaryMask(), BinaryMask2Volume(), SwapAxes([2, 1, 0])]
            ),
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                    SubsetSample(8192),
                    tgt.RandomFlip(axis=0),
                    tgt.RandomScale([0.8, 1.0]),
                    ThinPlateSplineAugmentation(),
                    tgt.RandomShear(0.05),
                    tgt.NormalizeScale(),
                    RandomRotation(sigma=0.2),
                    RandomTranslation(sigma=0.1),
                ]
            ),
        )
        model = pfm.RegularisedPointFlowSTNModel(
            n_components=512, fourier_feature_scale=0.2
        )

    elif experiment_name == "hippocampus":
        # INIT datasets
        dataset = MSDHippocampus(
            pre_transform=tgt.Compose(
                [
                    ThresholdImg2BinaryMask(threshold=0, components=None),
                    BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
                ]
            ),
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                    tgt.SamplePoints(8192),
                    ThinPlateSplineAugmentation(noise_sigma=0.1),
                    tgt.RandomShear(0.1),
                    tgt.NormalizeScale(),
                    RandomRotation(sigma=0.2),
                    RandomTranslation(sigma=0.1),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel(
            n_components=512, fourier_feature_scale=0.4
        )

    # elif experiment_name == "hippocampus_mln":
    #     # INIT datasets
    #     dataset = MSDHippocampus(
    #         pre_transform=tgt.Compose(
    #             [
    #                 ThresholdImg2BinaryMask(threshold=0, components=None),
    #                 BinaryMask2Surface(reduction_factor=None, pass_band=0.1),
    #             ]
    #         ),
    #         transform=tgt.Compose(
    #             [
    #                 tgt.NormalizeScale(),
    #                 tgt.SamplePoints(8192),
    #                 ThinPlateSplineAugmentation(noise_sigma=0.1),
    #                 tgt.RandomShear(0.1),
    #                 tgt.NormalizeScale(),
    #                 RandomRotation(sigma=0.2),
    #                 RandomTranslation(sigma=0.1),
    #             ]
    #         ),
    #     )

    #     model = pfm.RegularisedPointFlowSTNMLNModel(n_components=128)
    #     # model = pfm.RegularisedPointFlowSTNMLNModel(n_components=128)
    # elif experiment_name == "hippocampus_mln_hyper":
    #     # INIT datasets
    #     dataset = MSDHippocampus(
    #         pre_transform=tgt.Compose(
    #             [
    #                 # resample to lowest resolution
    #                 ThresholdImg2BinaryMask(threshold=0, components=None),
    #                 BinaryMask2Surface(reduction_factor=None),
    #             ]
    #         ),
    #         transform=tgt.Compose(
    #             [
    #                 tgt.NormalizeScale(),
    #                 tgt.SamplePoints(8192),
    #                 ThinPlateSplineAugmentation(noise_sigma=0.1),
    #                 tgt.RandomShear(0.1),
    #                 tgt.NormalizeScale(),
    #                 RandomRotation(sigma=0.2),
    #                 RandomTranslation(sigma=0.1),
    #             ]
    #         ),
    #     )

    #     model = pfm.RegularisedPointFlowSTNMLNModel(
    #         n_components=128, mln_metric="hyperbolic"
    #     )
    elif experiment_name == "brain":
        # INIT datasets
        dataset = ABCDBrain(
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                    tgt.SamplePoints(8192),
                    RandomRotation(sigma=0.2),
                    RandomTranslation(sigma=0.1),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel(
            n_components=512,
            fourier_feature_scale=0.8,
        )

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
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                    # tgt.SamplePoints(512),
                    tgt.SamplePoints(8192),
                    # tgt.SamplePoints(2 ** 16),
                    # ThinPlateSplineAugmentation(),
                    # tgt.RandomShear(0.05),
                    tgt.NormalizeScale(),
                    # RandomRotation(sigma=0.2),
                    # RandomTranslation(sigma=0.1),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel(
            #n_components=512,
            n_components=64,
            fourier_feature_scale=1.0,
            # fourier_feature_scale=0.25,
        )
    elif experiment_name == "gauss":
        # INIT datasets
        from gembed.dataset.synthetic_double_gaussian_dataset import (
            SyntheticGaussianDataset,
        )

        dataset = SyntheticGaussianDataset(
            n_samples=100,
            n_point_samples=8192,
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel(
            n_components=512,
            fourier_feature_scale=0.2,
        )

    elif experiment_name == "gauss2":
        # INIT datasets
        from gembed.dataset.synthetic_double_gaussian_dataset import (
            SyntheticGaussianDataset2,
        )

        dataset = SyntheticGaussianDataset2(
            n_samples=100,
            n_point_samples=8192,
            transform=tgt.Compose(
                [
                    tgt.NormalizeScale(),
                ]
            ),
        )

        model = pfm.RegularisedPointFlowSTNModel(
            n_components=512,
            fourier_feature_scale=0.4,
        )

    # elif experiment_name == "dental_debug":
    #     # INIT datasets
    #     from pittsburgh_dental_casts_debug import PittsburghDentalCastsDebug

    #     dataset = PittsburghDentalCastsDebug(
    #         pre_transform=tgt.Compose([SwapAxes([2, 0, 1]), InvertAxis(2)]),
    #         transform=tgt.Compose(
    #             [
    #                 tgt.NormalizeScale(),
    #                 tgt.SamplePoints(8192),
    #                 # RandomJitter(0.1),
    #                 RandomRotation(sigma=0.2),
    #                 RandomTranslation(sigma=0.1),
    #             ]
    #         ),
    #     )

    #     model = pfm.RegularisedPointFlowSTNModel(
    #         n_components=512, fourier_feature_scale=0.6
    #     )

    #     train_loader = DataLoader(
    #         dataset,
    #         shuffle=True,
    #         batch_size=4,
    #         num_workers=14,
    #         prefetch_factor=5,
    #         persistent_workers=True,
    #         pin_memory=True,
    #     )

    #     return model, train_loader, None

    else:
        raise ValueError("Experiment name not valid")

    # print("WARNING: selecting the first 10 datasamples only!!!")
    train, valid, test = train_valid_test_split(dataset)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    train_loader = DataLoader(
        # train[:1],
        train,
        shuffle=True,
        batch_size=1,
        # drop_last=True,  # drop the last batch since it might not be a complete batch
        num_workers=14,
        prefetch_factor=5,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid,
        shuffle=False,
        batch_size=20,
        num_workers=14,
        persistent_workers=True,
        pin_memory=True,
    )

    return model, train_loader, valid_loader


def train_model(experiment_name, model, train_loader, valid_loader):
    #  init training session
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
        save_dir=f"lightning_logs/{experiment_name}",
        name="point_flow",
    )

    checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")
    ckpt_default = ModelCheckpoint(dirpath=checkpoint_path)
    ckpt_intermediate_saves = ModelCheckpoint(
        save_top_k=-1,
        monitor="train_loss",
        dirpath=checkpoint_path,
        every_n_train_steps=10000,
        filename="intermediate_ckpt_{step}_{train_loss:.2f}",
    )

    print("WARNING: stopping criteria patience = 1e3")

    class StoppingCriteria(Callback):
        def __init__(self, patience=int(1e3), final_lr=1e-7, eps=1.01, **kwargs):
            super().__init__(**kwargs)
            self.patience = patience

            # eps is there to prevent numerical error
            self.final_lr = eps * final_lr

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            current_lr = next(iter(pl_module.optimizers().param_groups))["lr"]

            if current_lr <= self.final_lr:
                if self.patience <= 0:
                    trainer.should_stop = True
                    print("Model, converged! Stopping training session...")
                else:
                    self.patience -= 1

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="gpu",
        devices=[0],
        max_epochs=600000,
        log_every_n_steps=10,
        # callbacks=[ckpt_default, ckpt_intermediate_saves, StoppingCriteria()],
        callbacks=[ckpt_intermediate_saves, StoppingCriteria()],
        # detect_anomaly=True,
        gradient_clip_val=1e-3,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # val_dataloaders=valid_loader,
        # ckpt_path="lightning_logs/dental/point_flow/version_104/checkpoints/intermediate_ckpt_step=200000_train_loss=-1.32.ckpt",
        # ckpt_path=f"lightning_logs/hippocampus/point_flow/version_2/checkpoints/final_model.ckpt",
        # ckpt_path=f"lightning_logs/hippocampus/point_flow/version_1/checkpoints/intermediate_ckpt_step=10000_train_loss=-0.12.ckpt",
        # ckpt_path=f"lightning_logs/brain/point_flow/version_0/checkpoints/intermediate_ckpt_step=10000_train_loss=-0.00.ckpt",
        # ckpt_path=f"lightning_logs/skull/point_flow/version_1/checkpoints/intermediate_ckpt_step=30000_train_loss=-0.07.ckpt",
        # ckpt_path=f"lightning_logs/dental_curv/point_flow/version_3/checkpoints/intermediate_ckpt_step=40000_train_loss=-0.31.ckpt",
    )
    trainer.save_checkpoint(
        os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
    )


if __name__ == "__main__":
    experiment_name = sys.argv[-1]

    pl.seed_everything(42, workers=True)
    model, train_loader, valid_loader = load_config(experiment_name)
    train_model(experiment_name, model, train_loader, valid_loader)
