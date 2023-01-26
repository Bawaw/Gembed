#!/usr/bin/env python3

import torch
import unittest
from gembed.numerics.trace import trace, hutchinson_trace_estimator

class TraceTest(unittest.TestCase):

    def test_trace(self):
        x_in = torch.randn(100, 3).requires_grad_(True)
        x_out = x_in + 5

        TrJ, J = trace(x_out, x_in, return_jacobian = True)
        TrJ_2, J_2 = trace(x_out, x_in)

        epsilon = torch.randn(x_in.shape[0], 3)
        TrJ_estimate, J_prod = hutchinson_trace_estimator(x_out, x_in, epsilon)

        self.assertEqual(TrJ.shape, TrJ_estimate.shape)

if __name__ == "__main__":
    unittest.main()
