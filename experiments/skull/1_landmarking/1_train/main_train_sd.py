#!/usr/bin/env python3

import sys

import os
import gembed.models.point_score_diffusion_model as psd
from gembed.models import PointScoreDiffusionModel

from gembed.dataset import load_dataset
from gembed import Configuration
import torch_geometric.transforms as tgt
import lightning as pl
from gembed.dataset.paris_volumetric_skulls import ParisVolumetricSkulls
from gembed.utils.dataset import train_valid_test_split
from lightning.pytorch.strategies import DDPStrategy
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from gembed.transforms import RandomRotation, RandomTranslation, Clip
from math import isclose


def load_config(experiment_name):
    print(f"Loading experiment: {experiment_name}")

    # load dataset
    dataset = load_dataset(experiment_name)

    # load model
    model = PointScoreDiffusionModel.load(experiment_name)

    # holdout split
    train, valid, test = train_valid_test_split(dataset)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )
    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=45,  # 45,
        num_workers=15,
        prefetch_factor=30,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        # valid,
        train[[0, 7]],
        shuffle=False,
        batch_size=2,  # 6,
        num_workers=14,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    return model, train_loader, valid_loader


# def train_model(experiment_name, model, train_loader, valid_loader):
#     #  SETUP
#     tb_logger = pl.pytorch.loggers.TensorBoardLogger(
#         save_dir=f"lightning_logs/{experiment_name}",
#         name="score_diffusion",
#     )

#     checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")
#     ckpt_intermediate_saves = ModelCheckpoint(
#         save_top_k=-1,
#         monitor="train_point_loss",
#         dirpath=checkpoint_path,
#         every_n_train_steps=int(1e4),
#         filename="intermediate_ckpt_{step}_{train_loss:.2f}",
#     )

#     trainer = pl.Trainer(
#         logger=tb_logger,
#         accelerator="gpu",
#         devices=[0],
#         #max_steps=int(5e4),
#         max_steps=int(1e5),
#         val_check_interval=500,
#         check_val_every_n_epoch=None,
#         callbacks=[ckpt_intermediate_saves],
#         log_every_n_steps=1,
#         # detect_anomaly=True,
#         gradient_clip_val=1e-3,
#         gradient_clip_algorithm="norm",
#     )


#     # TRAIN POINT DIFFUSION
#     # model.set_phase(psd.Phase.TRAIN_POINT_DIFFUSION)

#     # trainer.fit(
#     #     model=model,
#     #     train_dataloaders=train_loader,
#     #     val_dataloaders=valid_loader,
#     #     #ckpt_path=f"lightning_logs/{experiment_name}/score_diffusion/version_2/checkpoints/point_diffusion_final_model.ckpt",
#     # )

#     # SAVE POINT DIFFUSION MODEL
#     # trainer.save_checkpoint(
#     #     os.path.join(
#     #         tb_logger.log_dir, "checkpoints", "point_diffusion_final_model.ckpt"
#     #     )
#     # )

#     ################################################
#     # model = psd.PointScoreDiffusionSTNModel.load_from_checkpoint(
#     #     "lightning_logs/hippocampus/score_diffusion/version_401/checkpoints/point_diffusion_final_model.ckpt",
#     #     n_components=512,  # n_components,
#     #     fourier_feature_scale_x=1.0,
#     #     fourier_feature_scale_t=30,
#     #     use_stn=False,
#     #     use_ltn=True,
#     #     lambda_kld=0,
#     # )

#     ################################################

#     # TRAIN LATENT DIFFUSION
#     # model.set_phase(psd.Phase.TRAIN_LATENT_DIFFUSION)
#     # if model.ltn is not None:

#     #     trainer = pl.Trainer(
#     #         logger=tb_logger,
#     #         accelerator="gpu",
#     #         devices=[0],
#     #         max_steps=int(5e4),
#     #         val_check_interval=500,
#     #         check_val_every_n_epoch=None,
#     #         log_every_n_steps=1,
#     #         # callbacks=[ckpt_intermediate_saves],
#     #         # detect_anomaly=True,
#     #         gradient_clip_val=1e-3,
#     #         gradient_clip_algorithm="norm",
#     #     )

#     #     trainer.fit(
#     #         model=model,
#     #         train_dataloaders=train_loader,
#     #     )

#     #     trainer.save_checkpoint(
#     #         os.path.join(
#     #             tb_logger.log_dir, "checkpoints", "latent_diffusion_final_model.ckpt"
#     #         )
#     #     )

#     model = psd.PointScoreDiffusionSTNModel.load_from_checkpoint(
#         "lightning_logs/hippocampus/score_diffusion/version_0/checkpoints/final_model.ckpt",
#         n_components=512,  # n_components,
#         fourier_feature_scale_x=1.0,
#         fourier_feature_scale_t=30,
#         use_stn=False,
#         use_ltn=True,
#         use_mtn=False,
#         lambda_kld=1e-8,
#     )
#     # ################################################

#     # model.mtn = psd.MetricTransformer(512)
#     model.mtn = psd.get_mtn()

#     # # TRAIN METRIC TRANSFORMER
#     model.set_phase(psd.Phase.TRAIN_METRIC_TRANSFORMER)

#     if model.mtn is not None:
#         trainer = pl.Trainer(
#             logger=tb_logger,
#             accelerator="gpu",
#             devices=[0],
#             max_steps=int(5e4),
#             val_check_interval=10,  # 500,
#             check_val_every_n_epoch=None,
#             log_every_n_steps=1,
#             # callbacks=[ckpt_intermediate_saves],
#             # detect_anomaly=True,
#             gradient_clip_val=1e-3,
#             gradient_clip_algorithm="norm",
#         )

#         trainer.fit(
#             model=model,
#             train_dataloaders=train_loader,
#             val_dataloaders=valid_loader,
#         )

#     trainer.save_checkpoint(
#         os.path.join(
#             tb_logger.log_dir, "checkpoints", "metric_transformer_final_model.ckpt"
#         )
#     )

#     # # SAVE FINAL MODEL
#     model.set_phase(psd.Phase.EVAL)
#     trainer.save_checkpoint(
#         os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
#     )


