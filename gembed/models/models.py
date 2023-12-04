# #!/usr/bin/env python3

# import os
# from glob import glob
# from gembed.models.point_score_diffusion_model import PointScoreDiffusionSTNModel
# import gembed.models.point_score_diffusion_model as psd

# import numpy as np

# def load_point_score_diffusion_model(experiment_name, version=None):
#     if experiment_name == "hippocampus":
#         model_kwargs = {
#             "n_components": 512,
#             "fourier_feature_scale_x": 1.0,
#             "fourier_feature_scale_t": 30,
#             "use_stn": False,
#             "use_ltn": True,
#             "use_mtn": True,
#             "lambda_kld": 1e-8,
#         }

#     elif experiment_name == "skull":
#         model_kwargs = {
#             "n_components": 512,
#             "fourier_feature_scale_x": 3.0,
#             "fourier_feature_scale_t": 30,
#             "use_stn": False,
#             "use_ltn": False,
#             "use_mtn": False,
#             "lambda_kld": 1e-8,
#             # SDM
#             "sdm_n_hidden_layers": 10,
#             "sdm_hidden_dim": 128,
#             # PDM
#             "pdm_n_hidden_layers": 20,
#             "pdm_hidden_dim": 128,
#             "pdm_hyper_hidden_dim": 128,
#         }
#     elif experiment_name == "brain":
#         model_kwargs = {
#             "n_components": 512,
#             "fourier_feature_scale_x": 3.0,
#             "fourier_feature_scale_t": 30,
#             "use_stn": False,
#             "use_ltn": False,
#             "use_mtn": False,
#             "lambda_kld": 1e-8,
#             # SDM
#             "sdm_n_hidden_layers": 10,
#             "sdm_hidden_dim": 128,
#             # PDM
#             "pdm_n_hidden_layers": 20,
#             "pdm_hidden_dim": 128,
#             "pdm_hyper_hidden_dim": 128,
#         }
#     elif experiment_name == "dental":
#         model_kwargs = {
#             "n_components": 512,
#             "fourier_feature_scale_x": 3.0,
#             "fourier_feature_scale_t": 30,
#             "use_stn": False,
#             "use_ltn": False,
#             "use_mtn": False,
#             "lambda_kld": 1e-8,
#             # SDM
#             "sdm_n_hidden_layers": 10,
#             "sdm_hidden_dim": 128,
#             # PDM
#             "pdm_n_hidden_layers": 20,
#             "pdm_hidden_dim": 128,
#             "pdm_hyper_hidden_dim": 128,
#         }
#     else:
#         raise ValueError(f"Invalid experiment name: {experiment_name}")

#     if version is None:
#         model = PointScoreDiffusionSTNModel(**model_kwargs)
#     else:
#         model_path = f"../1_train/lightning_logs/{experiment_name}/score_diffusion/version_{version}/checkpoints/final_model.ckpt"
#         if not os.path.exists(model_path):
#             model_path = glob(
#                 f"../1_train/lightning_logs/{experiment_name}/score_diffusion/version_{version}/checkpoints/epoch=*-step=*.ckpt"
#             )[-1]

#         print(f"Loading experiment: {experiment_name}, from path: {model_path}")
#         model = PointScoreDiffusionSTNModel.load_from_checkpoint(
#             model_path, **model_kwargs
#         )

#         model = model.eval()
#         model.set_phase(psd.Phase.EVAL)

#     return model
