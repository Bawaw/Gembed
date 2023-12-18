#!/usr/bin/env python3

import os
import torch
from glob import glob

import lightning as pl
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint

from gembed import Configuration
from gembed.core.module.point_flow import PointScoreDiffusion
from gembed.core.module.bijection import EncoderDecoder
from gembed.core.module.point_dynamics import (
    ResidualPointConditionalDynamics, ResidualPointDynamics)
from gembed.core.module.point_encoder import ResidualPCEncoder
from gembed.core.module.point_flow.point_score_diffusion import Phase
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.stochastic import VPSDE
from gembed.core.module.stochastic.variational_encoder import \
    VariationalEncoder
from gembed.models import ModelProtocol


class PointScoreDiffusionModel(PointScoreDiffusion, ModelProtocol):
    def __init__(
        self,
        # GLOBAL
        fourier_feature_scale_t,
        fourier_feature_scale_x,
        lambda_kld,
        n_components,

        # STN
        stn_fourier_feature_scale,
        stn_hidden_dim,
        stn_layer_type,
        stn_activation_type,
        stn_n_components,
        stn_n_hidden_layers,
        use_stn,
        
        # LTN
        ltn_activation_type,
        ltn_beta_max,
        ltn_beta_min,
        ltn_fourier_feature_scale,
        ltn_hidden_dim,
        ltn_hyper_hidden_dim,
        ltn_layer_type,
        ltn_n_hidden_layers,
        ltn_t_dim,
        use_ltn,

        # MTN
        mtn_hidden_dim,
        mtn_n_components,
        use_mtn,
        
        # SDM
        sdm_hidden_dim,
        sdm_layer_type,
        sdm_activation_type,
        sdm_n_hidden_layers,

        # PDM
        pdm_activation_type,
        pdm_beta_max,
        pdm_beta_min,
        pdm_hidden_dim,
        pdm_hyper_hidden_dim,
        pdm_in_channels,
        pdm_layer_type,
        pdm_n_hidden_layers,
        pdm_out_channels,
        pdm_t_dim,

        **kwargs
    ):
        # NETWORK CONFIG

        # spatial transformer
        if use_stn:
            stn = SpatialTransformer(
                ResidualPCEncoder(
                    n_components=stn_n_components,
                    fourier_feature_scale=stn_fourier_feature_scale,
                    n_hidden_layers=stn_n_hidden_layers,
                    hidden_dim=stn_hidden_dim,
                    layer_type=stn_layer_type,
                    activation_type=stn_activation_type,
                )
            )

        else:
            stn = None

        if use_ltn:
            ltn = VPSDE(
                beta_min=ltn_beta_min,
                beta_max=ltn_beta_max,
                dim=n_components,
                f_score=ResidualPointDynamics(
                    in_channels=n_components,
                    out_channels=n_components,
                    fourier_feature_scale_x=ltn_fourier_feature_scale,
                    fourier_feature_scale_t=fourier_feature_scale_t,
                    n_hidden_layers=ltn_n_hidden_layers,
                    hyper_hidden_dim=ltn_hyper_hidden_dim,
                    hidden_dim=ltn_hidden_dim,
                    t_dim=ltn_t_dim,
                    layer_type=ltn_layer_type,
                    activation_type=ltn_activation_type,
                ),
            )
        else:
            ltn = None

        if use_mtn:
            mtn = EncoderDecoder(
                    encoder = nn.Sequential(
                        nn.Linear(n_components, mtn_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(mtn_hidden_dim, mtn_n_components),
                    ),
                    decoder = nn.Sequential(
                        nn.Linear(mtn_n_components, mtn_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(mtn_hidden_dim, n_components),
                    )
                )

        else:
            mtn = None

        # shape distribution model
        sdm = VariationalEncoder(
            feature_nn=ResidualPCEncoder(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale_x,
                n_hidden_layers=sdm_n_hidden_layers,
                hidden_dim=sdm_hidden_dim,
                layer_type=sdm_layer_type,
                activation_type=sdm_activation_type,
            ),
            add_log_var_module=lambda_kld > 0,
        )

        # point distribution model
        pdm = VPSDE(
            beta_min=pdm_beta_min,
            beta_max=pdm_beta_max,
            dim=pdm_in_channels,
            f_score=ResidualPointConditionalDynamics(
                in_channels=pdm_in_channels,
                out_channels=pdm_out_channels,
                n_context=n_components,
                fourier_feature_scale_x=fourier_feature_scale_x,
                fourier_feature_scale_t=fourier_feature_scale_t,
                n_hidden_layers=pdm_n_hidden_layers,
                hyper_hidden_dim=pdm_hyper_hidden_dim,
                hidden_dim=pdm_hidden_dim,
                t_dim=pdm_t_dim,
                layer_type=pdm_layer_type,
                activation_type=pdm_activation_type,
            ),
        )

        super().__init__(
            sdm=sdm, pdm=pdm, stn=stn, ltn=ltn, mtn=mtn, lambda_kld=lambda_kld
        )

        # track model configuration
        self.save_hyperparameters()

    def fit(self, train_loader, valid_loader, model_name):
        root_dir = Configuration()["Paths"]["WEIGHT_DIR"]

        #  SETUP
        torch.set_float32_matmul_precision("high")
        tb_logger = pl.pytorch.loggers.TensorBoardLogger(
            save_dir=f"{root_dir}/{model_name}",
            name="score_diffusion",
        )

        checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")

        # TRAIN POINT DIFFUSION
        self.set_phase(Phase.TRAIN_POINT_DIFFUSION)
        
        # setup trainer
        ckpt_intermediate_saves = ModelCheckpoint(
            save_top_k=-1,
            monitor="train_point_loss",
            dirpath=checkpoint_path,
            every_n_train_steps=int(2e4),
            filename="phase_1_intermediate_ckpt_{step}_{train_point_loss:.2f}",
        )

        trainer = pl.Trainer(
            logger=tb_logger,
            accelerator="gpu",
            devices=[0],
            max_steps=int(2e5),
            val_check_interval=1000,
            check_val_every_n_epoch=None,
            callbacks=[ckpt_intermediate_saves],
            log_every_n_steps=1,
            gradient_clip_val=1e-3,
            gradient_clip_algorithm="norm",
        )

        # fit point diffusion model
        trainer.fit(
            model=self,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

        # save point diffusion model
        trainer.save_checkpoint(
            os.path.join(
                tb_logger.log_dir, "checkpoints", "phase_1_final_model.ckpt"
            )
        )

        # TRAIN LATENT DIFFUSION
        self.set_phase(Phase.TRAIN_LATENT_DIFFUSION)

        if self.ltn is not None:
            # setup trainer
            ckpt_intermediate_saves = ModelCheckpoint(
                save_top_k=-1,
                monitor="train_latent_loss",
                dirpath=checkpoint_path,
                every_n_train_steps=int(5e3),
                filename="phase_2_intermediate_ckpt_{step}_{train_latent_loss:.2f}",
            )

            trainer = pl.Trainer(
                logger=tb_logger,
                accelerator="gpu",
                devices=[0],
                max_steps=int(5e4),
                val_check_interval=500,
                check_val_every_n_epoch=None,
                log_every_n_steps=1,
                callbacks=[ckpt_intermediate_saves],
                gradient_clip_val=1e-3,
                gradient_clip_algorithm="norm",
            )

            # fit latent diffusion model
            trainer.fit(
                model=self,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )

            # save latent diffusion model
            trainer.save_checkpoint(
                os.path.join(
                    tb_logger.log_dir, "checkpoints", "phase_2_final_model.ckpt"
                )
            )

        # TRAIN METRIC TRANSFORMER
        self.set_phase(Phase.TRAIN_METRIC_TRANSFORMER)

        # setup trainer
        if self.mtn is not None:
            ckpt_intermediate_saves = ModelCheckpoint(
                save_top_k=-1,
                monitor="train_metric_loss",
                dirpath=checkpoint_path,
                every_n_train_steps=int(2e3),
                filename="phase_3_intermediate_ckpt_{step}_{train_metric_loss:.2f}",
            )

            trainer = pl.Trainer(
                logger=tb_logger,
                accelerator="gpu",
                devices=[0],
                max_steps=int(2e4),
                val_check_interval=200,
                check_val_every_n_epoch=None,
                log_every_n_steps=1,
                callbacks=[ckpt_intermediate_saves],
                gradient_clip_val=1e-3,
                gradient_clip_algorithm="norm",
            )

            # fit metric transformer model
            trainer.fit(
                model=self,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )

            # save metric transformer model
            trainer.save_checkpoint(
                os.path.join(
                    tb_logger.log_dir, "checkpoints", "phase_3_final_model.ckpt"
                )
            )

        # SAVE FINAL MODEL
        self.set_phase(Phase.EVAL)
        trainer.save_checkpoint(
            os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
        )

        return self

    def load(version=None, **model_kwargs):
        model_name = model_kwargs["model_name"]

        if version is None:
            model = PointScoreDiffusionModel(**model_kwargs)
        else:
            root_dir = Configuration()["Paths"]["WEIGHT_DIR"]

            model_path = f"{root_dir}/{model_name}/score_diffusion/version_{version}/checkpoints/final_model.ckpt"
            if not os.path.exists(model_path):
                model_path = glob(
                    f"{root_dir}/{model_name}/score_diffusion/version_{version}/checkpoints/phase_*_intermediate_ckpt_*_*.ckpt"
                )[-1]

            print(f"Loading experiment: {model_name}, from path: {model_path}")
            model = PointScoreDiffusionModel.load_from_checkpoint(
                model_path
            )

            model = model.eval()
            model.set_phase(Phase.EVAL)

        return model
