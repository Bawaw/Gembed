#!/usr/bin/env python3

import torch
import torch.nn as nn


# 111111111111111111111111111
class ConcatSquashLinear(nn.Module):
    """Source: https://github.com/stevenygd/PointFlow/blob/master/models/diffeq_layers.py"""

    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(ConcatSquashLinear, self).__init__()
        from torch.nn.utils.parametrizations import spectral_norm as sn

        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


# 222222222222222222222222222
class FusionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(FusionLayer, self).__init__()

        hidden_dim_x = 256
        self._proj_x = nn.Sequential(
            nn.Linear(dim_in, hidden_dim_x), nn.LayerNorm(hidden_dim_x), nn.Tanh()
        )

        hidden_dim_c = 256
        self._proj_c = nn.Sequential(
            nn.Linear(dim_c, hidden_dim_c), nn.LayerNorm(hidden_dim_c), nn.Tanh()
        )

        hidden_dim_t = 32
        self._proj_t = nn.Sequential(
            nn.Linear(32, hidden_dim_t), nn.LayerNorm(hidden_dim_t), nn.Tanh()
        )

        self._fuse = nn.Sequential(
            nn.Linear(hidden_dim_x + hidden_dim_c + hidden_dim_t, dim_out),
        )

        # self._attention = nn.Sequential(nn.Linear(dim_out, dim_out), nn.Softmax(dim=1))

    def forward(self, c, t, x):
        c = self._proj_c(c)
        x = self._proj_x(x)
        t = self._proj_t(t)

        x = self._fuse(torch.cat([c, t, x], 1))
        # scale = torch.tensor(x.shape[1])
        # w = self._attention(x / torch.sqrt(scale))

        return x


# 333333333333333333333333333
class TripleFusionLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(TripleFusionLayer, self).__init__()

        hidden_dim_x = 512
        self._proj_x = nn.Sequential(nn.Linear(512, 512), nn.Tanh())

        hidden_dim_c = 512
        self._proj_context = nn.Sequential(nn.Linear(512, 512), nn.Tanh())

        hidden_dim_t = 32
        self._proj_time = nn.Sequential(nn.Linear(32, 32), nn.Tanh())

        self._fuse = nn.Sequential(
            nn.Linear(hidden_dim_x + hidden_dim_c + hidden_dim_t, dim_out),
        )

    def forward(self, c, t, x):
        c = self._proj_context(c)
        t = self._proj_time(t)
        x = self._proj_x(x)

        x = self._fuse(torch.cat([c, t, x], 1))

        return x


# 444444444444444444444444444
class TripleFusionLayerAgg(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(TripleFusionLayerAgg, self).__init__()

        hidden_dim_x = 512
        self._proj_x = nn.Sequential(nn.Linear(512, 512), nn.Tanh())
        # self._attn_x = nn.Sequential(nn.Linear(512, 512), nn.Softmax(dim=1))
        self._attn_x = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        hidden_dim_c = 512
        self._proj_c = nn.Sequential(nn.Linear(512, 512), nn.Tanh())
        # self._attn_c = nn.Sequential(nn.Linear(512, 512), nn.Softmax(dim=1))
        self._attn_c = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        hidden_dim_t = 512
        self._proj_t = nn.Sequential(nn.Linear(32, 512), nn.Tanh())
        # self._attn_t = nn.Sequential(nn.Linear(512, 512), nn.Softmax(dim=1))
        self._attn_t = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

        self._fuse = nn.Sequential(
            nn.Linear(512, dim_out),
        )

    def forward(self, c, t, x):
        c = self._proj_c(c)
        t = self._proj_t(t)
        x = self._proj_x(x)

        # x = self._fuse(self._attn_x(c) * c + self._attn_t(t) * t + self._attn_x(x) * x)
        x = self._fuse(c + t + x)

        return x


# 555555555555555555555555555
# class TripleFusionTransformer(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
#         super(TripleFusionTransformer, self).__init__()

#         self._proj_x = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32, nhead=8), num_layers=3)

#         self._fuse = nn.Sequential(
#             nn.Linear(1056, dim_out),
#         )


#     def forward(self, c, t, x):
#         # B x N x C
#         c = c.view(-1, 16, 32)
#         x = x.view(-1, 16, 32)
#         t = t.view(-1, 1, 32)

#         X = torch.concat([c, x, t], -2)

#         X = self._proj_x(X).view(-1, 1056)
#         X = self._fuse(X)

#         return X

# class ConcatSquashLinear(nn.Module):
#     """Source: https://github.com/stevenygd/PointFlow/blob/master/models/diffeq_layers.py"""

#     def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
#         super(ConcatSquashLinear, self).__init__()

#         self._layer = nn.Linear(dim_in + dim_c + 1, dim_out)
#         self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

#     def forward(self, context, x):
#         x = torch.cat((x, context), dim=-1)
#         gate = torch.sigmoid(self._hyper_gate(context))

#         if x.dim() == 3:
#             gate = gate.unsqueeze(1)
#             bias = bias.unsqueeze(1)

#         ret = self._layer(x) * gate
#         return ret


# 66666666666666666666666666
class TripleFusionLayerMul(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
        super(TripleFusionLayerMul, self).__init__()

        hidden_dim = 512
        self._proj_x = nn.Sequential(nn.Linear(512, hidden_dim), nn.Tanh())
        self._proj_context = nn.Sequential(nn.Linear(512, hidden_dim), nn.Tanh())
        self._proj_time = nn.Sequential(nn.Linear(32, hidden_dim), nn.Tanh())

        self._fuse = nn.Sequential(
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, c, t, x):
        c = self._proj_context(c)
        t = self._proj_time(t)
        x = self._proj_x(x)

        x = self._fuse(c * t * x)

        return x


# class ConcatSquashLinear(nn.Module):
#     """Source: https://github.com/stevenygd/PointFlow/blob/master/models/diffeq_layers.py"""

#     def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
#         super(ConcatSquashLinear, self).__init__()

#         self._layer = nn.Linear(dim_in + dim_c + 1, dim_out)

#     def forward(self, context, x):
#         x = torch.cat((x, context), dim=-1)

#         ret = self._layer(x)
#         return ret


# class ConcatSquashLinear(nn.Module):
#     """Source: https://github.com/stevenygd/PointFlow/blob/master/models/diffeq_layers.py"""

#     def __init__(self, dim_in, dim_out, dim_c, zero_init=False):
#         super(ConcatSquashLinear, self).__init__()

#         self.layer_1 = nn.Linear(dim_in, int(dim_out / 2))
#         self.layer_2 = nn.Linear(dim_c + 1, int(dim_out / 2))
#         self.layer_3 = nn.Linear(dim_out, int(dim_out))

#     def forward(self, c, x):
#         x = torch.tanh(self.layer_1(x))
#         c = torch.tanh(self.layer_2(c))
#         ret = self.layer_3(torch.cat((x, c), dim=-1))
#         return ret
