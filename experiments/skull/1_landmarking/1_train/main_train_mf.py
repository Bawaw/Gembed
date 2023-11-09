#!/usr/bin/env python3


import sys

sys.path.insert(0, "../")

import os

import gembed.models.point_flow_model as pfm
import gembed.models.point_manifold_flow_model as pmfm
from gembed import Configuration
import torch_geometric.transforms as tgt
import pytorch_lightning as pl
from gembed.dataset import (
    MSDLiver,
    MSDHippocampus,
    ABCDBrain,
    PittsburghDentalCasts,
    PittsburghDentalCastsCurvature,
)
from gembed.core.module import Phase
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.utils.dataset import train_valid_test_split
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from gembed.transforms import RandomRotation, RandomTranslation, Clip
from transform import *
from math import isclose


def load_config(experiment_name):
    print(f"Loading experiment: {experiment_name}")

    if experiment_name == "dental":
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
                    tgt.SamplePoints(8192),
                    # ThinPlateSplineAugmentation(),
                    # tgt.RandomShear(0.05),
                    tgt.NormalizeScale(),
                    RandomRotation(sigma=0.2),
                    RandomTranslation(sigma=0.1),
                ]
            ),
        )

        model = pmfm.RegularisedPointManifoldFlowSTNModel(
            n_components=512,
            # fourier_feature_scale=2.0,
            #fourier_feature_scale=2.0,
            fourier_feature_scale=-1,
        )

    else:
        raise ValueError("Experiment name not valid")

    train, valid, test = train_valid_test_split(dataset)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    train_loader = DataLoader(
        # train,
        train[:10],
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
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
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

    class StoppingCriteria(Callback):
        def __init__(self, patience=int(1e4), final_lr=1e-7, eps=1.01, **kwargs):
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
        max_epochs=int(1e6),
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
    )

    model.set_phase(Phase.TRAIN_DENSITY_ESTIMATION)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # ckpt_path=f"lightning_logs/dental/point_flow/version_201/checkpoints/final_model.ckpt",
        # val_dataloaders=valid_loader,
    )

    trainer.save_checkpoint(
        os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
    )


if __name__ == "__main__":
    experiment_name = sys.argv[-1]

    pl.seed_everything(42, workers=True)
    model, train_loader, valid_loader = load_config(experiment_name)
    train_model(experiment_name, model, train_loader, valid_loader)
