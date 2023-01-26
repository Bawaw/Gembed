#!/usr/bin/env python3

import torch.nn as nn
from gembed.core.module import AbstractInvertibleModule

class Variational(AbstractInvertibleModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                z: Tensor,
                conditions: Union[Tensor, None] = None,
                batch: Union[Tensor, None] = None,
                include_log_px: bool = True):

        x = self.decoder(z, conditions, batch)
        if not include_log_px: return x

        # TODO: implement computation of log_px
        raise NotImplementedError()
        log_px = None

        return x, log_px

    def inverse(self,
                x: Tensor,
                conditions: Union[Tensor, None] = None
                batch: Union[Tensor, None] = None,
                include_log_px: bool = True):

        ds_z = ds_x.clone()
        log_qz, ds_z.state = self.encoder(ds_x.state, sample=sample)
        log_px, x_reconstr = self.decoder(ds_z.state, context=ds_x.state)

        # p(x) = p(z) + log p(x|z) - log q(z|x)
        ds_z.add_or_create('log_prob', log_px - log_qz)

        return ds_z
