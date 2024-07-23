"""Spatially structured grid classes."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import torch
from torch import Tensor

from torchbox3d.math.conversions import world_to_grid_coordinates
from torchbox3d.math.ops.cluster import ClusterType, cluster_grid


@dataclass
class RegularGrid:
    """Models a regular, n-dimensional grid.

    Reference: https://en.wikipedia.org/wiki/Regular_grid

    Args:
        min_world_coordinates_m: (N,) Minimum world coordinates (in meters).
        max_world_coordinates_m: (N,) Maximum world coordinates (in meters).
        delta_m_per_cell: (N,) Ratio of meters to cell in each dimension.
    """

    min_world_coordinates_m: Tuple[float, ...]
    max_world_coordinates_m: Tuple[float, ...]
    delta_m_per_cell: Tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the instance variables."""
        d_min = len(self.min_world_coordinates_m)
        d_max = len(self.max_world_coordinates_m)
        d_delta = len(self.delta_m_per_cell)

        if d_min not in (d_delta, d_max):
            raise ValueError(
                "`min_world_coordinates_m`, `max_world_coordinates_m` "
                "and `delta_m_per_cell` "
                "must have the same dimension!"
            )

    @property
    def num_dimensions(self) -> int:
        """Return number of dimensions in the grid."""
        return len(self.min_world_coordinates_m)

    @cached_property
    def grid_size(self) -> Tuple[int, ...]:
        """Return the size of the grid."""
        min_world_coordinates_m = torch.as_tensor(self.min_world_coordinates_m)
        max_world_coordinates_m = torch.as_tensor(self.max_world_coordinates_m)
        world_coordinates_range_m = max_world_coordinates_m - min_world_coordinates_m
        dimensions: List[int] = self.scale_and_center_coordinates(
            world_coordinates_range_m
        ).tolist()
        return tuple(dimensions)

    def scale_and_center_coordinates(
        self, coordinates_m: Tensor, align_corners: bool = True
    ) -> Tensor:
        """Scale and center the positions.

        Args:
            coordinates_m: (N,D) Coordinates in meters.
            align_corners: Boolean flag to treat indices as grid vertices.

        Returns:
            The scaled, centered positions.
        """
        num_dimensions = min(self.num_dimensions, coordinates_m.shape[-1])
        delta_m_per_cell = torch.as_tensor(
            self.delta_m_per_cell,
            device=coordinates_m.device,
            dtype=torch.float,
        )

        indices = (
            coordinates_m[..., :num_dimensions] / delta_m_per_cell[:num_dimensions]
        )
        if not align_corners:
            indices += 0.5
        return indices.long()

    def convert_world_coordinates_to_grid(
        self, coordinates_m: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Transform positions from world coordinates to grid coordinates (in meters).

        Args:
            coordinates_m: (N,D) Coordinates in meters.

        Returns:
            The grid indices and the cropped coordinate mask.
        """
        indices, mask = world_to_grid_coordinates(
            coordinates_m,
            list(self.min_world_coordinates_m),
            list(self.delta_m_per_cell),
            grid_size=list(self.grid_size),
        )
        return indices, mask

    def scale_grid(self, stride: int) -> Tuple[int, ...]:
        """Downsample the grid coordinates.

        Args:
            stride: Downsampling factor.

        Returns:
            The downsampled grid size.
        """
        grid_size = torch.as_tensor(self.grid_size)
        scaled_grid_size: List[int] = torch.ceil(grid_size / stride).int().tolist()
        return tuple(scaled_grid_size)

    @cached_property
    def grid_offset_m(self) -> Tuple[float, ...]:
        """Return the grid offset from the lower bound to the grid origin."""
        min_world_coordinates_m = [abs(x) for x in self.min_world_coordinates_m]
        return tuple(min_world_coordinates_m)

    def cluster(
        self,
        indices: Tensor,
        values: Tensor,
        cluster_type: ClusterType = ClusterType.MEAN,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Cluster a set of values by their respective positions.

        Args:
            indices: (N,D) Grid indices.
            values: (N,F) Grid values.
            cluster_type: Cluster type to be applied.

        Returns:
            The clustered grid indices, values, and counts.

        Raises:
            NotImplementedError: If the voxelization mode is not implemented.
        """
        return cluster_grid(
            indices=indices,
            values=values,
            grid_size=list(self.grid_size),
            cluster_type=cluster_type,
        )


@dataclass
class BEVGrid(RegularGrid):
    """Representation of a bird's-eye view grid."""

    min_world_coordinates_m: Tuple[float, float]
    max_world_coordinates_m: Tuple[float, float]
    delta_m_per_cell: Tuple[float, float]


@dataclass
class VoxelGrid(RegularGrid):
    """Representation of a voxel grid."""

    min_world_coordinates_m: Tuple[float, float, float]
    max_world_coordinates_m: Tuple[float, float, float]
    delta_m_per_cell: Tuple[float, float, float]
