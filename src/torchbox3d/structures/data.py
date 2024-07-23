"""Class for manipulation of 3D data and annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from torch import Tensor

from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.grid import RegularGrid
from torchbox3d.structures.meta import TensorStruct
from torchbox3d.structures.sparse_tensor import SparseTensor
from torchbox3d.structures.targets import GridTargets


@dataclass
class Data(TensorStruct):
    """General class for manipulating 3D data and associated annotations."""

    coordinates_m: Tensor
    counts: Tensor
    cuboids: Optional[Cuboids]
    uuids: Tuple[str, int]
    values: Tensor


@dataclass
class RegularGridData(Data):
    """Data encoded on a regular grid.

    Args:
        grid: Grid object.
        voxels: Sparse tensor object.
        targets: Target encodings.
    """

    grid: RegularGrid
    cells: Union[SparseTensor, Tensor]
    targets: Optional[GridTargets] = None
    range_image: Optional[Tensor] = None
