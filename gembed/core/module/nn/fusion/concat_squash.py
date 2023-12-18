
import torch

from torch import nn

class ConcatSquash(nn.Module):
    """The `ConcatSquash` class is a neural network module that applies a transformation $a * (x @ W.T)) + b$ to the
    input data. The weights $a, b$ are predicted by a hypernetworks."""
    
    def __init__(
        self,
        in_dim,
        out_dim,
        t_dim,
        hidden_dim,
        hyper_bias=None,
        hyper_gate=None,
        net=None,
    ):
        super().__init__()

        self.in_dim, self.out_dim, self.t_dim, self.hidden_dim = (
            in_dim,
            out_dim,
            t_dim,
            hidden_dim,
        )
        if hyper_bias is None:
            self.hyper_bias = nn.Sequential(
                nn.Linear(self.t_dim, self.hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_bias = hyper_bias

        if hyper_gate is None:
            self.hyper_gate = nn.Sequential(
                nn.Linear(self.t_dim, self.hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
        else:
            self.hyper_gate = hyper_gate

        if net is None:
            self.net = nn.Linear(self.in_dim, self.out_dim, bias=False)
        else:
            self.net = net

    def forward(self, x, t, batch):
        n_batch = batch.max() + 1

        # get weight matrices
        a = torch.sigmoid(self.hyper_gate(t.view(n_batch, self.t_dim)))
        b = self.hyper_bias(t.view(n_batch, self.t_dim))

        # evaluate layer: (a * (x @ W.T)) + b
        result = a * self.net(x) + b
        return result
