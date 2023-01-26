#!/usr/bin/env python3

import torch
from torch import Tensor
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from gembed.registration.abstract_registration_algorithm import (
    AbstractRegistrationAlgorithm,
)
from gembed.core.module.bijection import LinearTransformation


class AffineRegistration(AbstractRegistrationAlgorithm):
    def _estimate_transform(self, fixed: Data, moving: Data) -> Tensor:
        m_pos, f_pos = moving.pos, fixed.pos

        # 4x4 matrix to allow for bias
        # https://i.stack.imgur.com/aDm4s.jpg
        _moving = torch.cat([m_pos, torch.ones(m_pos.shape[0], 1).to(m_pos.device)], -1)
        _fixed = torch.cat([f_pos, torch.ones(f_pos.shape[0], 1).to(f_pos.device)], -1)

        # least squares solution
        return torch.linalg.lstsq(_moving, _fixed)[0]

    def _transform(self, moving: Data, param: Tensor) -> Data:
        f_star = moving.clone()
        m_pos = moving.pos

        m_pos = torch.cat([m_pos, torch.ones(m_pos.shape[0], 1).to(m_pos.device)], -1)

        transform = LinearTransformation(param)
        f_star.pos = transform(m_pos)[..., :-1]

        return f_star
