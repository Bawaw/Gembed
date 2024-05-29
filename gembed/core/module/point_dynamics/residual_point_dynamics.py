import torch
import torch_geometric.nn as tgnn
from gembed.core.module.nn.residual import ResidualCoefficient
from gembed.core.module.nn.spectral import FourierFeatureMap
from gembed.core.module.nn.fusion import (ConcatSquash, LinearCombination)
import torch.nn as nn

class ResidualPointDynamics(nn.Module):
    """The `ResidualPointDynamics` class is a neural network module that models the dynamics of a point
    cloud by applying residual layers with various activation functions and regression layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        fourier_feature_scale_x,
        fourier_feature_scale_t,
        n_hidden_layers,
        hyper_hidden_dim,
        hidden_dim,
        t_dim,
        layer_type,
        activation_type,
    ):
        super().__init__()
        self.activation_type = activation_type

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
            self.ffm_t = nn.Linear(1, t_dim)
        else:
            self.ffm_t = FourierFeatureMap(1, t_dim, fourier_feature_scale_t)

        def layer(in_channels, out_channels):
            if layer_type == "linear_combination":
                return LinearCombination(in_channels, t_dim, out_channels)
            elif layer_type == "concatsquash":
                return ConcatSquash(
                    in_channels,
                    out_channels,
                    t_dim,
                    hyper_hidden_dim,
                )

        def activation():
            if self.activation_type == "sin":
                return torch.sin
            elif self.activation_type == "selu":
                return nn.SELU()
            elif self.activation_type == "tanh":
                return nn.Tanh()
            elif self.activation_type == "tanhshrink":
                return nn.Tanhshrink()
            elif self.activation_type == "softplus":
                return nn.Softplus()
            elif self.activation_type == "swish":
                return nn.SiLU()
            elif self.activation_type == "elu":
                return nn.ELU()
            elif self.activation_type == "relu":
                return nn.ReLU()
            else:
                assert False

        self.nn_x = layer(hidden_dim, hidden_dim)

        # hidden layers
        self.layers = nn.ModuleList(
            [
                tgnn.Sequential(
                    "x, t, batch",
                    [
                        (activation(),"x -> x"),
                        nn.BatchNorm1d(hidden_dim),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
                        activation(),
                        nn.BatchNorm1d(hidden_dim),
                        (
                            layer(hidden_dim, hidden_dim),
                            "x, t, batch -> x",
                        ),
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
                activation(),
                # L2
                (
                    layer(hidden_dim, hidden_dim),
                    "x,t,batch -> x",
                ),
                activation(),
                # L3
                (
                    layer(hidden_dim, out_channels),
                    "x,t,batch -> x",
                ),
            ],
        )

    def forward(self, x, t, batch, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        if t.dim() == 0:
            t = t.repeat(batch.max() + 1, 1)

        # prep input
        x, t = self.ffm_x(x), self.ffm_t(t)

        for i, f in enumerate(self.layers):
            if i == 0: 
                x = x + f(self.nn_x(x, t, batch=batch), t, batch=batch)
            else:
                x = x + f(x, t, batch=batch)

        # return velocity
        return self.regression(x, t, batch=batch)
