#!/usr/bin/env python3

from torch import Tensor
from gembed.core.module import InvertibleModule


class LinearTransformation(InvertibleModule):
    """The `LinearTransformation` module represents a linear transformation that applies a
    fixed transformation matrix to an input tensor.
    """

    def __init__(self, transformation_matrix: Tensor):
        """
        The function initializes an object with a transformation matrix.
        
        :param transformation_matrix: The transformation_matrix parameter is a tensor that represents a
        transformation matrix. A transformation matrix describes how to transform a
        vector from one coordinate system to another and is used in transformations such as 
        translation, rotation, scaling,

        :type transformation_matrix: Tensor
        """
        super().__init__()
        self.M = transformation_matrix

    def forward(self, z: Tensor, **kwargs) -> Tensor:
        """
        The function performs a forward pass by multiplying the input tensor `z` with a matrix `M`.
        
        :param z: The parameter `z` is a tensor that represents the input 
        :return: the result of multiplying the input tensor `z` with the instance's transformation matrix.
        """
        return z @ self.M

    def inverse(self, x: Tensor, **kwargs) -> Tensor:
        # TODO: make this efficient based on the type of transform
        raise NotImplementedError()
