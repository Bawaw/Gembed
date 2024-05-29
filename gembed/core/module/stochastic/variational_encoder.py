#!/usr/bin/env python3

from copy import deepcopy

import lightning as pl
import torch

from gembed.core.module import InvertibleModule


class VariationalEncoder(pl.LightningModule, InvertibleModule):
    """The VariationalEncoder class is a module that performs variational encoding using a
    neural network, the regression part of the provided encoder will be duplicated to estimate the variance.
    """

    def __init__(self, feature_nn, add_log_var_module, batch_norm_mean=False): 
        super().__init__()

        self.feature_nn = feature_nn
        self.batch_norm_mean = batch_norm_mean

        # apply batch norm to the mean, this can help prevent posterior collapse
        if batch_norm_mean: 
            self.bn = torch.nn.BatchNorm1d(feature_nn.regression[-1].out_features, affine=False)

        if add_log_var_module:
            # add log var regression module for VAEs
            self.log_var_regression = deepcopy(self.feature_nn.regression)

    def get_params(self, x, batch):
        embedding = self.feature_nn.feature_forward(x, batch)

        mean = self.feature_nn.regression(embedding)
        log_var = self.log_var_regression(embedding)

        if self.batch_norm_mean:
            mean = self.bn(mean)

        return mean, log_var

    def inverse(self, x, batch, stochastic=False):
        if stochastic:
            Z_mean, Z_log_var = self.get_params(x, batch=batch)
            Z_std = torch.exp(0.5 * Z_log_var)
            return Z_mean + Z_std * torch.randn_like(Z_mean)
        else:
            if self.batch_norm_mean:
                return self.bn(self.feature_nn(x, batch))
            else:
                return self.feature_nn(x, batch)