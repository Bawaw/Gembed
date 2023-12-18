import json
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
from math import isclose


def load_config(config_name):
    print(f"Starting training for model: {config_name}")

    # load dataset
    dataset = load_dataset(config_name)

    # load model
    path_model_kwargs = os.path.join(Configuration()["Paths"]["MODEL_CONFIG_DIR"], config_name + ".json")
    
    model = PointScoreDiffusionModel.load(
        **json.load(open(path_model_kwargs))
    )

    # holdout split
    train, valid, test = train_valid_test_split(dataset)

    print(
        f"Total number of scans: {len(dataset)}, split in {len(train)}/{len(valid)}/{len(test)}"
    )
    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=45,  
        num_workers=15,
        prefetch_factor=30,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid,
        shuffle=False,
        batch_size=2,  # 6,
        num_workers=14,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    return model, train_loader, valid_loader

if __name__ == "__main__":
    # e.g. bash$ main_train.py hippocampus
    config_name = sys.argv[-1]

    pl.seed_everything(42, workers=True)
    model, train_loader, valid_loader = load_config(config_name)

    model.fit(train_loader, valid_loader, config_name)
