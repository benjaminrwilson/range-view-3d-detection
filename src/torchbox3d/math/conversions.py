"""Geometric conversions."""

from typing import List, Tuple, Union

import torch
from torch import Tensor

from torchbox3d.math.crop import crop_coordinates
from torchbox3d.math.ops.cluster import ClusterType, cluster_grid


@torch.jit.script
def normalized_to_denormalized_intensities(tensor: Tensor) -> Tensor:
    """Map intensities from [0,1] -> [0,255].

    Args:
        tensor: (...,K) K-channel tensor image.

    Returns:
        (...,K) The uint8 tensor image.
    """
    tensor = tensor.clamp(0, 1)
    tensor *= 255.0
    tensor = tensor.byte()
    return tensor


@torch.jit.script
def cartesian_to_spherical_coordinates(
    coordinates_cartesian_m: Tensor,
) -> Tensor:
    """Convert Cartesian coordinates to spherical coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        coordinates_cartesian_m: Cartesian coordinates (x,y,z) in meters.

    Returns:
        (N,3) Spherical coordinates (azimuth,inclination,radius).
    """
    coordinates_x = coordinates_cartesian_m[..., 0]
    coordinates_y = coordinates_cartesian_m[..., 1]
    coordinates_z = coordinates_cartesian_m[..., 2]

    hypot_xy = coordinates_x.hypot(coordinates_y)
    radius = hypot_xy.hypot(coordinates_z)
    inclination = coordinates_z.atan2(hypot_xy)
    azimuth = coordinates_y.atan2(coordinates_x)
    coordinates_spherical = torch.stack((azimuth, inclination, radius), dim=-1)
    return coordinates_spherical


@torch.jit.script
def spherical_to_cartesian_coordinates(
    coordinates_spherical: Tensor,
) -> Tensor:
    """Convert spherical coordinates to Cartesian coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        coordinates_spherical: Spherical coordinates (azimuth,inclination,radius).

    Returns:
        (N,3) Cartesian coordinates (x,y,z).
    """
    coordinates_azimuth = coordinates_spherical[..., 0]
    coordinates_inclination = coordinates_spherical[..., 1]
    coordinates_radius = coordinates_spherical[..., 2]

    rcos_inclination = coordinates_radius * coordinates_inclination.cos()
    coordinates_x = rcos_inclination * coordinates_azimuth.cos()
    coordinates_y = rcos_inclination * coordinates_azimuth.sin()
    coordinates_z = coordinates_radius * coordinates_inclination.sin()
    cartesian_coordinates_m = torch.stack(
        (coordinates_x, coordinates_y, coordinates_z), dim=-1
    )
    return cartesian_coordinates_m


# @torch.jit.script
def world_to_grid_coordinates(
    coordinates_m: Tensor,
    min_world_coordinates_m: Union[List[float], Tensor],
    delta_m_per_cell: Union[List[float], Tensor],
    grid_size: Union[List[int], Tensor],
    align_corners: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert world coordinates to grid coordinates.

    Args:
        coordinates_m: (N,D) Coordinates in meters.
        min_world_coordinates_m: (D,) Minimum coordinates in meters.
        delta_m_per_cell: (D,) Ratio of meters to cell in each dimension.
        grid_size: (D,) Size of the grid.
        align_corners: Boolean flag to align the world coordinates to the cell
            centers.

    Returns:
        The grid coordinates and the cropped points mask.
    """
    if isinstance(min_world_coordinates_m, List):
        min_world_coordinates_m = torch.as_tensor(
            min_world_coordinates_m, device=coordinates_m.device
        )
    if isinstance(delta_m_per_cell, List):
        delta_m_per_cell = torch.as_tensor(
            delta_m_per_cell, device=coordinates_m.device
        )
    if isinstance(grid_size, List):
        grid_size = torch.as_tensor(grid_size, device=coordinates_m.device)

    num_dimensions = min(coordinates_m.shape[-1], len(min_world_coordinates_m))
    offset_m = torch.zeros_like(coordinates_m[0])
    offset_m[:num_dimensions] = torch.as_tensor(
        min_world_coordinates_m[:num_dimensions]
    ).abs()

    indices = (
        coordinates_m[..., :num_dimensions] + offset_m[:num_dimensions]
    ) / delta_m_per_cell[:num_dimensions]
    if not align_corners:
        indices[..., :num_dimensions] += 0.5
    indices = indices.long()

    upper = [float(x) for x in grid_size]
    _, mask = crop_coordinates(indices, [0.0, 0.0, 0.0], upper)
    return indices, mask


def voxelize(
    coordinates_m: Tensor,
    values: Tensor,
    min_world_coordinates_m: List[float],
    delta_m_per_cell: List[float],
    grid_size: List[int],
    align_corners: bool = False,
    max_num_values: int = 1,
    cluster_type: ClusterType = ClusterType.MEAN,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Voxelize a set of coordinates and values.

    Args:
        coordinates_m: (N,D) Coordinates in meters.
        values: (N,F) Values at each coordinate.
        min_world_coordinates_m: (D,) Minimum coordinates in meters.
        delta_m_per_cell: (D,) Ratio of meters to cell in each dimension.
        grid_size: (D,) Size of the grid.
        align_corners: Boolean flag to align the world coordinates to the cell
            centers.
        max_num_values: Max number of values per cell.
        cluster_type: Type of clustering to perform.

    Returns:
        The voxelized indices, values, and the counts per voxel.
    """
    indices, mask = world_to_grid_coordinates(
        coordinates_m,
        min_world_coordinates_m,
        delta_m_per_cell,
        grid_size,
        align_corners,
    )

    indices, values, counts = cluster_grid(
        indices, values, grid_size, max_num_values, cluster_type
    )
    return indices, values, counts


def BCHW_to_BKC(x: Tensor) -> Tensor:
    """Convert BCHW to B(K=H*W)C format.

    Args:
        x: Input tensor (BCHW).

    Returns:
        Reshaped tensor (BKC) where K = H * W.
    """
    num_batches, num_channels, _, _ = x.shape
    x = x.permute(0, 2, 3, 1)
    x = x.view(num_batches, -1, num_channels)
    return x
