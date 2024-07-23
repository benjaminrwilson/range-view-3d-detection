"""Clustering operations."""

from enum import Enum, unique
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torchbox3d.math.ops.index import ravel_multi_index, unravel_index


@unique
class ClusterType(str, Enum):
    """The type of reduction performed during voxelization."""

    CONCATENATE = "CONCATENATE"
    MEAN = "MEAN"


def cluster_grid(
    indices: Tensor,
    values: Tensor,
    grid_size: List[int],
    max_num_values: int = 20,
    cluster_type: ClusterType = ClusterType.MEAN,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a clustering operation on a grid.

    Args:
        indices: (N,3) Grid indices.
        values: (N,F) Grid values.
        grid_size: (3,) Length, width, and height of the grid.
        max_num_values: int = 1,
        cluster_type: ClusterType = ClusterType.MEAN,

    Returns:
        The spatial indices, features, and counts after clustering.

    Raises:
        NotImplementedError: If the clustering type is not implemented.
    """
    if cluster_type.upper() == ClusterType.MEAN:
        indices, values, counts = _mean_cluster_grid_kernel(indices, values, grid_size)
    elif cluster_type.upper() == ClusterType.CONCATENATE:
        indices, values, counts = _concatenate_cluster_grid_kernel(
            indices,
            values,
            grid_size,
            max_num_values=max_num_values,
        )
    else:
        raise NotImplementedError("This clustering type is not implemented!")
    return indices, values, counts


@torch.jit.script
def _mean_cluster_grid_kernel(
    indices: Tensor, values: Tensor, grid_size: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a mean clustering operation on a grid.

    Args:
        indices: (N,3) Spatial indices.
        values: (N,F) Spatial values.
        grid_size: (3,) Length, width, and height of the grid.

    Returns:
        The spatial indices, features, and counts after clustering.
    """
    raveled_indices = ravel_multi_index(indices, grid_size)

    # CPU Fallback due to MPS limitations.
    is_mps = raveled_indices.device.type == "mps"
    if is_mps:
        permutation_sorted = torch.argsort(raveled_indices.cpu()).to("mps")
    else:
        permutation_sorted = torch.argsort(raveled_indices)

    raveled_indices = raveled_indices[permutation_sorted]
    values = values[permutation_sorted]

    # Compute unique values, inverse, and counts.
    out: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        raveled_indices, return_inverse=True, return_counts=True
    )
    output, inverse_indices, counts = out

    # Scatter with sum reduction.
    # CPU Fallback due to MPS limitations.
    if is_mps:
        last_index_exclusive = int(inverse_indices.cpu().max()) + 1
    else:
        last_index_exclusive = int(inverse_indices.max()) + 1
    pooled_values: Tensor = torch.zeros_like(values).scatter_add_(
        dim=0,
        index=inverse_indices[:, None].repeat(1, values.shape[-1]),
        src=values,
    )[:last_index_exclusive]
    pooled_values /= counts[:, None]

    # Unravel unique linear indices into unique multi indices.
    unraveled_coords = unravel_index(output, grid_size)

    out_inv: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        inverse_indices,
        return_inverse=True,
        return_counts=True,
    )
    offset, _, counts = out_inv
    offset_csum = F.pad(counts[:-1] - 1, pad=[1, 0], mode="constant", value=0.0)

    # CPU Fallback due to MPS limitations.
    if is_mps:
        offset_csum = offset_csum.cpu().cumsum(dim=0).to("mps")
    else:
        offset_csum = offset_csum.cumsum(dim=0)

    offset += offset_csum

    # Respect original ordering.
    # CPU Fallback due to MPS limitations.
    if is_mps:
        inverse_indices = torch.argsort(permutation_sorted[offset].cpu()).to("mps")
    else:
        inverse_indices = torch.argsort(permutation_sorted[offset])
    return (
        unraveled_coords[inverse_indices],
        pooled_values[inverse_indices],
        counts[inverse_indices],
    )


@torch.jit.script
def _concatenate_cluster_grid_kernel(
    indices: Tensor,
    values: Tensor,
    grid_size: List[int],
    max_num_values: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply a concatenation clustering operation on a grid.

    NOTE: This will not pool the points that fall into a bin. Instead,
    this function will concatenate the points until they exceed a maximium
    size defined by max_num_pts.

    Args:
        indices: (N,3) Spatial indices.
        values: (N,F) Spatial values.
        grid_size: (3,) Length, width, and height of the grid.
        max_num_values: Max number of values per index location.

    Returns:
        The spatial indices, values, counts, and cropping mask.
    """
    raveled_indices = ravel_multi_index(indices, grid_size)

    # Find indices which make bucket indices contiguous.
    permutation_sorted = torch.argsort(raveled_indices)
    indices = indices[permutation_sorted]
    raveled_indices = raveled_indices[permutation_sorted]
    values = values[permutation_sorted]

    # Compute unique values, inverse, and counts.
    out: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        raveled_indices, return_inverse=True, return_counts=True
    )

    output, inverse_indices, counts = out

    # Initialize vectors at each voxel (max_num_pts,F).
    # Instead of applying an information destroying reduction,
    # we concatenate the features until we reach a maximum size.
    voxelized_values = torch.zeros((len(output), max_num_values, values.shape[-1]))

    # Concatenating collisions requires counting how many collisions there are.
    # This computes offsets for all of the collisions in a vectorized fashion.
    offset = F.pad(counts, pad=[1, 0], mode="constant", value=0.0).cumsum(dim=0)[
        inverse_indices
    ]

    index = torch.arange(0, len(inverse_indices)) - offset
    out_inv: Tuple[Tensor, Tensor, Tensor] = torch.unique_consecutive(
        inverse_indices,
        return_inverse=True,
        return_counts=True,
    )
    offset, _, _ = out_inv
    is_valid = index < max_num_values

    inverse_indices = inverse_indices[is_valid]
    index = index[is_valid].long()
    voxelized_values[inverse_indices, index] = values[is_valid]

    inv_perm = torch.argsort(permutation_sorted[offset])
    voxelized_indices = unravel_index(output, grid_size).int()
    voxelized_indices = voxelized_indices[inv_perm]
    voxelized_values = voxelized_values[inv_perm]
    counts = counts[inv_perm]
    return voxelized_indices, voxelized_values, counts
