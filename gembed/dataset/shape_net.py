import re
import os

from gembed import Configuration
from lightning import LightningDataModule
from torch_geometric.datasets import ShapeNet as ShapeNetTG

class ShapeNet(ShapeNetTG, LightningDataModule):
    def __init__(
        self,
        root=None,
        categories=None,
        **kwargs
    ):
        if root is None:
            path = Configuration()["Paths"]["DATA_DIR"]
            root = os.path.join(path, self.subdir)

        super().__init__(
            root=root,
            categories=categories, 
            include_normals=False,
            **kwargs
        )

    @property
    def subdir(self) -> str:
        return re.sub('((?!^)(?<!_)[A-Z][a-z]+|(?<=[a-z0-9])[A-Z])', r'_\1', self.__class__.__name__).lower()