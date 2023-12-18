#!/usr/bin/env python3


import sys

sys.path.insert(0, "../")

import os

import gembed.models.linear_point_flow as lpf
from gembed import Configuration
import torch_geometric.transforms as tgt
import lightning as pl
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.utils.dataset import train_valid_test_split
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from gembed.dataset import LowResParisPCASkulls
from gembed.transforms import RandomRotation, RandomTranslation, Clip
from transform import (
    ExcludeIDs,
    SubsetSample,
    SwapAxes,
    ThresholdImg2BinaryMask,
    BinaryMask2Surface,
    BinaryMask2Volume,
    ThinPlateSplineAugmentation,
)

def load_config(experiment_name):
    print(f"Loading experiment: {experiment_name}")

    if experiment_name == "pca_surface":
        N_COMPONENTS = 28

        # INIT datasets
        dataset = LowResParisPCASkulls(
            pre_transform=tgt.NormalizeScale(),
            transform=tgt.Compose([
                tgt.SamplePoints(4096)
            ]),
            n_samples=100,
            affine_align=True,
            n_components=N_COMPONENTS,
        )

        model = lpf.SingleLaneLPF(n_components=N_COMPONENTS)

    else:
        raise ValueError("Experiment name not valid")

    train, valid, test = train_valid_test_split(dataset)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )

    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=10,
        num_workers=7,
        prefetch_factor=5,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid,
        shuffle=False,
        batch_size=20,
        num_workers=3,
        persistent_workers=True,
    )

    return model, train_loader, valid_loader

def train_model(experiment_name, model, train_loader, valid_loader):
    #  init training session
    tb_logger = pl.pytorch.loggers.TensorBoardLogger(
        save_dir=f"lightning_logs/{experiment_name}",
        name="point_flow",
    )
    checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")
    topk_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="valid_loss",
        mode="min",
        dirpath=checkpoint_path,
        filename="{step:02d}-{valid_loss:.2f}",
    )
    default_checkpoint = ModelCheckpoint(dirpath=checkpoint_path)

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator="gpu",
        #devices=[1, 2, 3, 4, 6, 7],
        devices=[0],
        # devices=[1, 2],
        max_epochs=60000,
        deterministic=True,
        log_every_n_steps=5,  # 1 log per epoch
        callbacks=[default_checkpoint, topk_checkpoint],
        #strategy=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    trainer.save_checkpoint(
        os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
    )


if __name__ == "__main__":
    experiment_name = sys.argv[-1]

    pl.seed_everything(42, workers=True)
    model, train_loader, valid_loader = load_config(experiment_name)
    train_model(experiment_name, model, train_loader, valid_loader)
