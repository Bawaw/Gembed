import torch
import pytorch_lightning as pl
from torch_scatter import scatter_mean, scatter_sum
from pytorch_lightning.utilities import grad_norm
from enum import Enum
from gembed.core.module import InvertibleModule
from .point_flow import PointFlowSTN


class Phase(Enum):
    TRAIN_MANIFOLD_LEARNING = 1
    TRAIN_DENSITY_ESTIMATION = 2
    EVAL = 3


class ManifoldFlowWrapper(InvertibleModule):
    def __init__(self, manifold_flow, ambient_flow):
        super().__init__()
        self.manifold_flow = manifold_flow
        self.ambient_flow = ambient_flow

    def forward(self, z, include_combined_dynamics=False, **kwargs):

        x0, *combined_dynamics = self.ambient_flow.forward(
            z=z, include_combined_dynamics=include_combined_dynamics, **kwargs
        )


        x1, _ = self.manifold_flow.forward(
            z=x0, include_combined_dynamics=include_combined_dynamics, **kwargs
        )

        return x1, *combined_dynamics

    def inverse(self, x, include_combined_dynamics=False, **kwargs):

        z0, *_ = self.manifold_flow.inverse(
            x=x, include_combined_dynamics=include_combined_dynamics, **kwargs
        )

        z1, *combined_dynamics = self.ambient_flow.inverse(
            x=z0, include_combined_dynamics=include_combined_dynamics, **kwargs
        )

        return z1, *combined_dynamics


class RegularisedPointManifoldFlowSTN(PointFlowSTN):
    def __init__(self, phase=None, lambda_e=1e-3, lambda_n=1e-3, lambda_m=0, **kwargs):
        super(RegularisedPointManifoldFlowSTN, self).__init__(**kwargs)
        if phase is None:
            phase = Phase.TRAIN_MANIFOLD_LEARNING
        self.set_phase(phase)
        self.lambda_e = lambda_e
        self.lambda_n = lambda_n
        self.lambda_m = lambda_m

    def eval(self):
        self.phase = Phase.EVAL
        return super().eval()

    def set_phase(self, phase):
        self.phase = phase

        if phase == Phase.TRAIN_MANIFOLD_LEARNING:
            self.sdm.freeze(False)
            self.pdm.layers.ambient_flow.freeze(True)
            self.pdm.layers.manifold_flow.freeze(False)

        elif phase == Phase.TRAIN_DENSITY_ESTIMATION:
            self.sdm.freeze(True)
            self.pdm.layers.manifold_flow.freeze(True)
            self.pdm.layers.ambient_flow.freeze(False)

        else:
            self.sdm.freeze(False)
            self.pdm.layers.manifold_flow.freeze(False)
            self.pdm.layers.ambient_flow.freeze(False)

    def MLLoss(self, train_batch, batch_idx, split="train"):
        x, batch = train_batch.pos, train_batch.batch

        if self.stn is not None:
            x = self.stn(x, batch)

        n_samples_per_example = 512#torch.bincount(batch)

        condition = self.inverse(x, batch=batch)

        x = x[:n_samples_per_example, :]
        batch = batch[:n_samples_per_example]

        z, _, kinetic_energy = self.pdm.layers.manifold_flow.inverse(
            x=x,
            batch=batch,
            condition=condition,
            include_combined_dynamics=True,
        )
        x_rec, _, kinetic_energy_2 = self.pdm.layers.manifold_flow.forward(
            z=z,
            batch=batch,
            condition=condition,
            include_combined_dynamics=True,
        )

        # PLOT #
        # from gembed.vis import plot_objects
        # plot_objects(
        #     (x.cpu(), None),
        #     (z.cpu(), None),
        #     (x_rec.cpu(), None),
        # )
        ########

        rec_error = (x - x_rec).pow(2).sum(-1)

        rec_error, kinetic_energy = scatter_sum(
            torch.stack([rec_error, kinetic_energy], -1), batch, dim=0
        ).T

        REC = rec_error / n_samples_per_example
        KE = kinetic_energy / n_samples_per_example

        REC = REC / x.shape[1]
        KE = self.lambda_e * (KE / x.shape[1])

        loss = REC + KE

        REC, KE, loss = REC.mean(), KE.mean(), loss.mean()

        self.log(f"ML_{split}_rec", REC, batch_size=train_batch.num_graphs)
        self.log(f"ML_{split}_ke", KE, batch_size=train_batch.num_graphs)
        self.log(f"ML_{split}_loss", loss, batch_size=train_batch.num_graphs)

        return loss

    def DELoss(self, train_batch, batch_idx, split="train"):
        x, batch = train_batch.pos, train_batch.batch

        if self.stn is not None:
            x = self.stn(x, batch)

        n_samples_per_example = torch.bincount(batch)

        condition = self.inverse(x, batch=batch)

        _, log_px, kinetic_energy, norm_jacobian = self.pdm.inverse(
            x,
            batch=batch,
            condition=condition,
            include_combined_dynamics=True,
            include_log_density=True,
        )

        log_px, kinetic_energy, norm_jacobian = scatter_sum(
            torch.stack([log_px, kinetic_energy, norm_jacobian], -1), batch, dim=0
        ).T

        NLL = -(log_px / n_samples_per_example)
        KE = kinetic_energy / n_samples_per_example
        NJ = norm_jacobian / n_samples_per_example

        NLL = NLL / x.shape[1]
        KE = self.lambda_e * (KE / x.shape[1])
        NJ = self.lambda_n * (NJ / x.shape[1])

        loss = NLL + KE + NJ

        NLL, KE, NJ, loss = NLL.mean(), KE.mean(), NJ.mean(), loss.mean()

        self.log(f"DE_{split}_nll", NLL, batch_size=train_batch.num_graphs)
        self.log(f"DE_{split}_ke", KE, batch_size=train_batch.num_graphs)
        self.log(f"DE_{split}_nj", NJ, batch_size=train_batch.num_graphs)
        self.log(f"DE_{split}_loss", loss, batch_size=train_batch.num_graphs)
        return loss

    @property
    def loss(self):
        if self.phase == Phase.TRAIN_MANIFOLD_LEARNING:
            return self.MLLoss
        elif self.phase == Phase.TRAIN_DENSITY_ESTIMATION:
            return self.DELoss
        else:
            raise ValueError("Manifold Point Flow in invalid phase for training.")

    def training_step(self, train_batch, batch_idx):
        return self.loss(train_batch, batch_idx, "train")

    def validation_step(self, valid_batch, batch_idx):
        return self.loss(train_batch, batch_idx, "valid")

    def configure_optimizers(self):

        if self.phase == Phase.TRAIN_MANIFOLD_LEARNING:
            optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)

            optim_dict = {"optimizer": optimiser}

            # plateau scheduler
            optim_dict["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    patience=int(1e4),
                    verbose=True,
                    min_lr=1e-7
                    # optimiser, patience=int(1e3), verbose=True, min_lr=1e-7
                ),
                "monitor": "ML_train_loss",
                "interval": "step",
            }
        else:
            optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)

            optim_dict = {"optimizer": optimiser}

            optim_dict["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    patience=int(1e4),
                    verbose=True,
                    min_lr=1e-7
                    # optimiser, patience=int(1e3), verbose=True, min_lr=1e-7
                ),
                "monitor": "DE_train_loss",
                "interval": "step",
            }

        return optim_dict
