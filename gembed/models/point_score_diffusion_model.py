#!/usr/bin/env python3

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from gembed.core.module.stn import SpatialTransformer
from gembed.core.module.mln import ContinuousDGM
from gembed.core.distribution import MultivariateNormal
from gembed.nn.fusion import HyperConcatSquash, LinearCombination, ConcatFuse, ConcatSquash
from copy import deepcopy
from gembed.core.module import PointScoreDiffusion, InvertibleModule
from gembed.core.module.bijection import AbstractODE
from gembed.core.module.stochastic import VPSDE, SubVPSDE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from gembed.models import ModelProtocol
from gembed.core.module.bijection import (
    ContinuousAmbientFlow,
    RegularisedContinuousAmbientFlow,
    EncoderDecoder
)
from glob import glob
from gembed.nn.linear.concat_squash_linear import *
from gembed.core.module.spectral import FourierFeatureMap
from gembed.core.module.regression import (
    ResidualRegressionModule,
    ResidualDGCNRegressionModule,
)
from gembed.nn.residual import ResidualCoefficient
from scipy import integrate

from gembed.core.module.point_score_diffusion import Phase
from gembed import Configuration

class FCResidual(nn.Module):
    r""" Models the latent dynamics."""

    def __init__(
        self,
        hidden_dim=512,
        n_hidden_layers=5,
    ):
        super().__init__()

        # hidden layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    ResidualCoefficient(),
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        for f in self.layers:
            x = x + f(x)

        return self.regression(x)


class LatentFDyn(nn.Module):
    r""" Models the latent dynamics."""

    def __init__(
        self,
        in_channels,
        out_channels,
        fourier_feature_scale_t,
        n_hidden_layers=3,
        hyper_hidden_dim=512,
        hidden_dim=512,
        t_dim=32,
        layer_type="concatsquash",
        activation_type="swish",
    ):
        super().__init__()

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        self.ffm_x = nn.Linear(in_channels, hidden_dim)

        if fourier_feature_scale_t is None:
            self.ffm_t = nn.Linear(1, 32)
        else:
            self.ffm_t = FourierFeatureMap(1, 32, fourier_feature_scale_t)

        def layer(in_channels, out_channels):
            if layer_type == "linear_combination":
                return LinearCombination(in_channels, 32, out_channels)
            elif layer_type == "concatsquash":
                return ConcatSquash(
                    in_channels,
                    out_channels,
                    t_dim,
                    hyper_hidden_dim,
                )

        def activation():
            if activation_type == "softplus":
                return nn.Softplus()
            elif activation_type == "swish":
                return nn.SiLU()

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, t, batch",
                    [
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
                        # nn.LayerNorm(hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                        activation(),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
                        # (nn.LayerNorm(hidden_dim), "x -> x"),
                        ResidualCoefficient(),
                    ],
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, t, batch",
            [
                # # L1
                (
                    layer(hidden_dim, hidden_dim),
                    "x,t,batch -> x",
                ),
                nn.LayerNorm(hidden_dim),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L2
                (
                    layer(hidden_dim, hidden_dim),
                    "x,t,batch -> x",
                ),
                (nn.LayerNorm(hidden_dim), "x -> x"),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L3
                (
                    layer(hidden_dim, out_channels),
                    "x,t,batch -> x",
                ),
                ResidualCoefficient(),
            ],
        )

    def forward(self, x, t, batch, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if t.dim() == 0:
            t = t.repeat(batch.max() + 1, 1)

        # prep input
        x, t = self.ffm_x(x), self.ffm_t(t)

        for f in self.layers:
            x = x + f(x, t, batch=batch)

        # return velocity
        return self.regression(x, t, batch=batch)


class FDyn(nn.Module):
    r""" Models the dynamics."""

    def __init__(
        self,
        n_context,
        fourier_feature_scale_x,
        fourier_feature_scale_t,
        in_channels=3,
        n_hidden_layers=3,
        hyper_hidden_dim=512,
        hidden_dim=512,
        t_dim=32,
        out_channels=3,
    ):
        super().__init__()

        # small scale = large kernel (underfitting)
        # large scale = small kernel (overfitting)
        print(f"FFS_x: {fourier_feature_scale_x}, FFS_t: {fourier_feature_scale_t}")
        if fourier_feature_scale_x is None:
            self.ffm_x = nn.Linear(in_channels, hidden_dim)
        else:
            self.ffm_x = FourierFeatureMap(
                in_channels, hidden_dim, fourier_feature_scale_x
            )

        if fourier_feature_scale_t is None:
            self.ffm_t = nn.Linear(1, 32)
        else:
            self.ffm_t = FourierFeatureMap(1, 32, fourier_feature_scale_t)

        layer_type = "hyperconcatsquash"
        # layer_type = "linear_combination"
        # activation_type = "softplus"
        # activation_type = "tanh"
        activation_type = "swish"

        def layer(in_channels, out_channels):
            if layer_type == "hyperconcatsquash":
                return HyperConcatSquash(
                    in_channels,
                    out_channels,
                    n_context,
                    t_dim,
                    hyper_hidden_dim,
                )
            elif layer_type == "linear_combination":
                return LinearCombination(in_channels, n_context, t_dim, out_channels)

            elif layer_type == "concat_fuse":
                return ConcatFuse(in_channels, n_context, t_dim, out_channels)
            else:
                assert False

        def activation():
            if activation_type == "tanh":
                return nn.Tanh()
            elif activation_type == "tanhshrink":
                return nn.Tanhshrink()
            elif activation_type == "softplus":
                return nn.Softplus()
            elif activation_type == "swish":
                return nn.SiLU()
            else:
                assert False

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, c, t, batch",
                    [
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, c, t, batch -> x",
                        ),
                        # nn.LayerNorm(hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                        activation(),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, c, t, batch -> x",
                        ),
                        # (nn.LayerNorm(hidden_dim), "x -> x"),
                        ResidualCoefficient(),
                    ],
                )
                for _ in range(n_hidden_layers)
            ]
        )

        # final regression layer
        self.regression = tgnn.Sequential(
            "x, c, t, batch",
            [
                # # L1
                (
                    layer(hidden_dim, hidden_dim),
                    "x,c,t,batch -> x",
                ),
                nn.LayerNorm(hidden_dim),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L2
                (
                    layer(hidden_dim, hidden_dim),
                    "x,c,t,batch -> x",
                ),
                (nn.LayerNorm(hidden_dim), "x -> x"),
                # (nn.BatchNorm1d(hidden_dim), "x -> x"),
                activation(),
                # L3
                (
                    layer(hidden_dim, out_channels),
                    "x,c,t,batch -> x",
                ),
                ResidualCoefficient(),
            ],
        )

    def forward(self, x, t, c, batch, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        # prep input
        x, t = self.ffm_x(x), self.ffm_t(t)

        for f in self.layers:
            x = x + f(x, c, t, batch=batch)

        # return velocity
        return self.regression(x, c, t, batch=batch)


class ShapeModel(InvertibleModule):
    def __init__(self, feature_nn, add_log_var_module=False):
        super().__init__()
        self.feature_nn = feature_nn

        if add_log_var_module:
            # add log var regression module for VAEs
            self.log_var_regression = deepcopy(self.feature_nn.regression)

    def get_params(self, pos, batch):
        embedding = self.feature_nn.feature_forward(pos, batch)

        mean = self.feature_nn.regression(embedding)
        log_var = self.log_var_regression(embedding)

        return mean, log_var

    def inverse(self, pos, batch, return_params=False):
        return self.feature_nn(pos, batch)

    def __str__(self):
        return str(self.__class__.str())


class MetricTransformer(InvertibleModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, Z, **kwargs):
        return self.model.forward(Z, **kwargs)

    def inverse(self, X, **kwargs):
        return self.model.inverse(X, **kwargs)

    def __str__(self):
        return str(self.__class__.str())


class PointScoreDiffusionModel(PointScoreDiffusion, ModelProtocol):
    def __init__(
        self,
        n_components=128,
        k=20,
        lambda_kld=1e-8,
        aggr="max",
        integration_method="rk4",
        fourier_feature_scale_x=1.0,
        fourier_feature_scale_t=1.0,
        beta_max=3,
        beta_min=1e-4,
        ltn_n_hidden_layers=5,
        ltn_hidden_dim=512,
        ltn_hyper_hidden_dim=128,
        sdm_n_hidden_layers=3,
        sdm_hidden_dim=128,
        pdm_n_hidden_layers=10,
        pdm_hidden_dim=128,
        pdm_hyper_hidden_dim=128,
        adjoint=False,
        use_stn=True,
        use_ltn=True,
        use_mtn=True,
    ):
        # NETWORK CONFIG

        # spatial transformer
        if use_stn:
            stn = SpatialTransformer(
                ResidualRegressionModule(
                    n_components=2 * 3,
                    fourier_feature_scale=-1,
                    n_hidden_layers=2,
                    hidden_dim=64,
                    layer_type="pointnet",
                )
            )

        else:
            stn = None

        if use_ltn:
            ltn = VPSDE(
                # pdm = SubVPSDE(
                beta_min=beta_min,
                beta_max=8,
                f_score=LatentFDyn(
                    in_channels=n_components,
                    out_channels=n_components,
                    fourier_feature_scale_t=fourier_feature_scale_t,
                    n_hidden_layers=ltn_n_hidden_layers,
                    hidden_dim=ltn_hidden_dim,
                    hyper_hidden_dim=ltn_hyper_hidden_dim,
                ),
            )
        else:
            ltn = None

        if use_mtn:
            mtn = get_mtn()
        else:
            mtn = None

        # shape distribution model
        sdm = ShapeModel(
            feature_nn=ResidualRegressionModule(
                n_components=n_components,
                fourier_feature_scale=fourier_feature_scale_x,
                n_hidden_layers=sdm_n_hidden_layers,
                hidden_dim=sdm_hidden_dim,
                layer_type="pointnet",
            ),
            add_log_var_module=lambda_kld > 0,
        )

        # point distribution model
        pdm = VPSDE(
            beta_min=beta_min,
            beta_max=beta_max,
            f_score=FDyn(
                n_context=n_components,
                fourier_feature_scale_x=fourier_feature_scale_x,
                fourier_feature_scale_t=fourier_feature_scale_t,
                n_hidden_layers=pdm_n_hidden_layers,
                hidden_dim=pdm_hidden_dim,
                hyper_hidden_dim=pdm_hyper_hidden_dim,
                in_channels=3,
                out_channels=3,
            ),
        )

        super().__init__(
            sdm=sdm, pdm=pdm, stn=stn, ltn=ltn, mtn=mtn, lambda_kld=lambda_kld
        )

    def fit(self, train_loader, valid_loader, model_name=None):
        root_dir = Configuration()["Paths"]["MODEL_DIR"]

        #  SETUP
        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=f"{root_dir}/{model_name}",
            name="score_diffusion",
        )

        checkpoint_path = os.path.join(tb_logger.log_dir, "checkpoints")
        ckpt_intermediate_saves = ModelCheckpoint(
            save_top_k=-1,
            monitor="train_point_loss",
            dirpath=checkpoint_path,
            every_n_train_steps=int(1e4),
            filename="phase_1_intermediate_ckpt_{step}_{train_point_loss:.2f}",
        )

        trainer = pl.Trainer(
            logger=tb_logger,
            accelerator="gpu",
            devices=[0],
            #max_steps=int(5e4),
            max_steps=int(1e5),
            val_check_interval=500,
            check_val_every_n_epoch=None,
            callbacks=[ckpt_intermediate_saves],
            log_every_n_steps=1,
            # detect_anomaly=True,
            gradient_clip_val=1e-3,
            gradient_clip_algorithm="norm",
        )
        # TRAIN POINT DIFFUSION
        # self.set_phase(Phase.TRAIN_POINT_DIFFUSION)

        # trainer.fit(
        #     model=self,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=valid_loader,
        #     #ckpt_path=f"{root_dir}/{model_name}/score_diffusion/version_2/checkpoints/point_diffusion_final_model.ckpt",
        # )

        # SAVE POINT DIFFUSION MODEL
        # trainer.save_checkpoint(
        #     os.path.join(
        #         tb_logger.log_dir, "checkpoints", "point_diffusion_final_model.ckpt"
        #     )
        # )

        ################################################
        # self = PointScoreDiffusionSTNModel.load_from_checkpoint(
        #     f"{root_dir}/hippocampus/score_diffusion/version_401/checkpoints/point_diffusion_final_model.ckpt",
        #     n_components=512,  # n_components,
        #     fourier_feature_scale_x=1.0,
        #     fourier_feature_scale_t=30,
        #     use_stn=False,
        #     use_ltn=True,
        #     lambda_kld=0,
        # )

        ################################################

        # TRAIN LATENT DIFFUSION
        # model.set_phase(Phase.TRAIN_LATENT_DIFFUSION)
        # if model.ltn is not None:
        # ckpt_intermediate_saves = ModelCheckpoint(
        #     save_top_k=-1,
        #     monitor="train_latent_loss",
        #     dirpath=checkpoint_path,
        #     every_n_train_steps=int(1e4),
        #     filename="phase_2_intermediate_ckpt_{step}_{train_latent_loss:.2f}",
        # )

        #     trainer = pl.Trainer(
        #         logger=tb_logger,
        #         accelerator="gpu",
        #         devices=[0],
        #         max_steps=int(5e4),
        #         val_check_interval=500,
        #         check_val_every_n_epoch=None,
        #         log_every_n_steps=1,
        #         # callbacks=[ckpt_intermediate_saves],
        #         # detect_anomaly=True,
        #         gradient_clip_val=1e-3,
        #         gradient_clip_algorithm="norm",
        #     )

        #     trainer.fit(
        #         model=model,
        #         train_dataloaders=train_loader,
        #     )

        #     trainer.save_checkpoint(
        #         os.path.join(
        #             tb_logger.log_dir, "checkpoints", "latent_diffusion_final_model.ckpt"
        #         )
        #     )

        self = PointScoreDiffusionModel.load_from_checkpoint(
            f"{root_dir}/hippocampus/score_diffusion/version_0/checkpoints/final_model.ckpt",
            n_components=512,  # n_components,
            fourier_feature_scale_x=1.0,
            fourier_feature_scale_t=30,
            use_stn=False,
            use_ltn=True,
            use_mtn=False,
            lambda_kld=1e-8,
        )
        # ################################################

        self.mtn = get_mtn()

        # # TRAIN METRIC TRANSFORMER
        self.set_phase(Phase.TRAIN_METRIC_TRANSFORMER)

        ckpt_intermediate_saves = ModelCheckpoint(
            save_top_k=-1,
            monitor="train_metric_loss",
            dirpath=checkpoint_path,
            every_n_train_steps=int(1e3),
            filename="phase_3_intermediate_ckpt_{step}_{train_metric_loss:.2f}",
        )

        if self.mtn is not None:
            trainer = pl.Trainer(
                logger=tb_logger,
                accelerator="gpu",
                devices=[0],
                max_steps=int(5e4),
                val_check_interval=10,  # 500,
                check_val_every_n_epoch=None,
                log_every_n_steps=1,
                callbacks=[ckpt_intermediate_saves],
                # detect_anomaly=True,
                gradient_clip_val=1e-3,
                gradient_clip_algorithm="norm",
            )

            trainer.fit(
                model=self,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )

        trainer.save_checkpoint(
            os.path.join(
                tb_logger.log_dir, "checkpoints", "metric_transformer_final_model.ckpt"
            )
        )

        # # SAVE FINAL MODEL
        self.set_phase(Phase.EVAL)
        trainer.save_checkpoint(
            os.path.join(tb_logger.log_dir, "checkpoints", "final_model.ckpt")
        )

        return self

    def load(model_name, version=None):

        if model_name == "hippocampus":
            model_kwargs = {
                "n_components": 512,
                "fourier_feature_scale_x": 1.0,
                "fourier_feature_scale_t": 30,
                "use_stn": False,
                "use_ltn": True,
                "use_mtn": True,
                "lambda_kld": 1e-8,
            }

        elif model_name == "skull":
            model_kwargs = {
                "n_components": 512,
                "fourier_feature_scale_x": 3.0,
                "fourier_feature_scale_t": 30,
                "use_stn": False,
                "use_ltn": False,
                "use_mtn": False,
                "lambda_kld": 1e-8,
                # SDM
                "sdm_n_hidden_layers": 10,
                "sdm_hidden_dim": 128,
                # PDM
                "pdm_n_hidden_layers": 20,
                "pdm_hidden_dim": 128,
                "pdm_hyper_hidden_dim": 128,
            }
        elif model_name == "brain":
            model_kwargs = {
                "n_components": 512,
                "fourier_feature_scale_x": 3.0,
                "fourier_feature_scale_t": 30,
                "use_stn": False,
                "use_ltn": False,
                "use_mtn": False,
                "lambda_kld": 1e-8,
                # SDM
                "sdm_n_hidden_layers": 10,
                "sdm_hidden_dim": 128,
                # PDM
                "pdm_n_hidden_layers": 20,
                "pdm_hidden_dim": 128,
                "pdm_hyper_hidden_dim": 128,
            }
        elif model_name == "dental":
            model_kwargs = {
                "n_components": 512,
                "fourier_feature_scale_x": 3.0,
                "fourier_feature_scale_t": 30,
                "use_stn": False,
                "use_ltn": False,
                "use_mtn": False,
                "lambda_kld": 1e-8,
                # SDM
                "sdm_n_hidden_layers": 10,
                "sdm_hidden_dim": 128,
                # PDM
                "pdm_n_hidden_layers": 20,
                "pdm_hidden_dim": 128,
                "pdm_hyper_hidden_dim": 128,
            }
        else:
            raise ValueError(f"Invalid experiment name: {model_name}")

        if version is None:
            model = PointScoreDiffusionModel(**model_kwargs)
        else:
            root_dir = Configuration()["Paths"]["MODEL_DIR"]

            model_path = f"{root_dir}/{model_name}/score_diffusion/version_{version}/checkpoints/final_model.ckpt"
            if not os.path.exists(model_path):
                model_path = glob(
                    f"{root_dir}/{model_name}/score_diffusion/version_{version}/checkpoints/phase_*_intermediate_ckpt_*_*.ckpt"
                )[-1]

            print(f"Loading experiment: {model_name}, from path: {model_path}")
            model = PointScoreDiffusionModel.load_from_checkpoint(
                model_path, **model_kwargs
            )

            model = model.eval()
            model.set_phase(Phase.EVAL)

        return model

def get_mtn():
    print("TODO: remove this")
    mtn = MetricTransformer(
        EncoderDecoder(
            encoder = nn.Sequential(
                nn.Linear(512, 256),
            #    nn.ReLU(),
                nn.SiLU(),
                nn.Linear(256, 128),
            ),
            decoder = nn.Sequential(
                nn.Linear(128, 256),
            #    nn.ReLU(),
                nn.SiLU(),
                nn.Linear(256, 512),
            )
            # encoder = nn.Sequential(
            #     nn.Linear(512, 256),
            #     #nn.ReLU(),
            #     nn.SiLU(),
            #     nn.Linear(256, 256),
            #     #,n.ReLU(),
            #     nn.SiLU(),
            #     nn.Linear(256, 128),
            # ),
            # decoder = nn.Sequential(
            #     nn.Linear(128, 256),
            #     #nn.ReLU(),
            #     nn.SiLU(),
            #     nn.Linear(256, 256),
            #     #nn.ReLU(),
            #     nn.SiLU(),
            #     nn.Linear(256, 512),
            # )
        )
    )

    return mtn
