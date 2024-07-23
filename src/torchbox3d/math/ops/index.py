"""Methods for manipulating indices."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def ravel_multi_index(unraveled_coords: Tensor, shape: List[int]) -> Tensor:
    """Convert a tensor of flat indices in R^K into flattened coordinates in R^1.

    'Untangle' a set of spatial indices in R^K to a 'flattened'
        or 'raveled' set of indices in R^1.

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html
    Reference:
        https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py#L18

    Args:
        unraveled_coords: (N,K) Tensor containing the indices.
        shape: (K,) Shape of the target grid.

    Returns:
        (N,) Tensor of 1D indices.
    """
    shape_tensor = torch.as_tensor(
        shape + [1],
        device=unraveled_coords.device,
        dtype=unraveled_coords.dtype,
    )

    # CPU Fallback due to MPS limitations.
    is_mps = unraveled_coords.device.type == "mps"
    if len(unraveled_coords) > 0:
        if is_mps:
            unraveled_coords = unraveled_coords.cpu()
            shape_tensor = shape_tensor.cpu()

        max_coords, _ = unraveled_coords.max(dim=0)
        assert torch.all(max_coords < shape_tensor[:-1])

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    raveled_coords = torch.mul(unraveled_coords, coefs).sum(dim=-1)

    # Move back to MPS.
    if is_mps:
        raveled_coords = raveled_coords.to("mps")
    return raveled_coords


@torch.jit.script
def unravel_index(raveled_indices: Tensor, shape: List[int]) -> Tensor:
    """Convert a tensor of flat indices in a multi-index of coordinate indices.

    'Tangle' a set of spatial indices in R^1 to an 'aligned' or 'unraveled'
    set of indices in R^K where K=len(dims).

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html.
    Reference:
        https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py#L37

    Args:
        raveled_indices: (N,) Tensor whose elements are indices
            into the flattened version of an array of dimensions shape.
        shape: (K,) List of dimensions.

    Returns:
        (N,K) Tensor of unraveled coordinates where K=len(dims).
    """
    shape_tensor = torch.as_tensor(
        shape,
        device=raveled_indices.device,
        dtype=raveled_indices.dtype,
    )

    # CPU Fallback due to MPS limitations.
    is_mps = raveled_indices.device.type == "mps"
    if is_mps:
        raveled_indices = raveled_indices.cpu()
        shape_tensor = shape_tensor.cpu()

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_ones((1,))), dim=0)
    unraveled_coords = (
        torch.div(raveled_indices[..., None], coefs, rounding_mode="trunc")
        % shape_tensor
    )

    # Move back to MPS.
    if is_mps:
        unraveled_coords = unraveled_coords.to("mps")
    return unraveled_coords


@torch.jit.script
def scatter_nd(
    index: Tensor,
    src: Tensor,
    grid_shape: List[int],
    permutation: Optional[List[int]] = None,
) -> Tensor:
    """Emplace (scatter) a set of values at the index locations.

    Args:
        index: (N,K) Tensor of coordinates.
        src: (N,K) Values to emplace.
        grid_shape: (K,) Size of each dimension.
        permutation: Permutation to apply after scattering.

    Returns:
        The scattered output.
    """
    if src.ndim == 1:
        src = src[:, None]

    raveled_indices = ravel_multi_index(index, grid_shape[:-1])[:, None].repeat(
        1, grid_shape[-1]
    )
    dst = torch.zeros(grid_shape, device=src.device, dtype=src.dtype)
    dst.view(-1, grid_shape[-1]).scatter_add_(dim=0, index=raveled_indices, src=src)
    if permutation is not None:
        dst = dst.permute(permutation)
    return dst


@torch.jit.script
def mgrid(intervals: List[List[int]]) -> Tensor:
    """Construct a meshgrid from a list of intervals.

    NOTE: Variable args are not used here to maintain JIT support.
    TODO: Explore rewrite with variadic args.

    Args:
        intervals: List of list of intervals.

    Returns:
        The constructed meshgrid.
    """
    tensor_list = [torch.arange(start, end) for start, end in intervals]
    meshgrid = torch.meshgrid(tensor_list, indexing="ij")
    return torch.stack(meshgrid, dim=0)


@torch.jit.script
def ogrid(intervals: List[List[int]]) -> Tensor:
    """Return a sparse multi-dimensional 'meshgrid'.

    Generate the Cartesian product of the intervals,
        represented as a sparse tensor.

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html

    Args:
        intervals: Any number of integer intervals.

    Returns:
        The sparse representation of the meshgrid.
    """
    tensor_list = [torch.arange(start, end) for start, end in intervals]
    meshgrid = torch.meshgrid(tensor_list, indexing="ij")
    sparse_meshgrid = torch.stack(meshgrid, dim=-1).view(-1, len(intervals))
    return sparse_meshgrid


@torch.jit.script
def ogrid_symmetric(intervals: List[int]) -> Tensor:
    """Compute a _symmetric_ sparse multi-dimensional 'meshgrid'.

    Unlike `ogrid` this function does specify start and stop positions
        for the indices. Instead, coordinates are centered about the
        origin.

    Reference:
        https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html

    Args:
        intervals: Any number of integer intervals.

    Returns:
        The sparse, _symmetric_ representation of the meshgrid.
    """
    lowers = [i // 2 for i in intervals]
    uppers = [i - low for i, low in zip(intervals, lowers)]
    symmetric_intervals = [[-low, u] for low, u in zip(lowers, uppers)]
    symmetric_ogrid: Tensor = ogrid(symmetric_intervals)
    return symmetric_ogrid


@torch.jit.script
def ogrid_sparse_neighborhoods(offsets: Tensor, intervals: List[int]) -> Tensor:
    """Compute a sparse representation of multiple meshgrids.

    Args:
        offsets: (N,K) Tensor representing the "center" of each
            sparse meshgrid.
        intervals: (K,) List of symmetric neighborhoods to consider.

    Returns:
        (N,K) The tensor containing all of the sparse neighborhoods.

    Raises:
        ValueError: If the per-offset dimension doesn't match the length
            of the intervals.
    """
    if not offsets.shape[-1] == len(intervals):
        raise ValueError(
            "The per-offset dimension and the length "
            "of the sparse intervals _must_ match."
        )
    ogrid = ogrid_symmetric(intervals)
    ogrid_sparse: Tensor = (offsets[..., None, :] + ogrid[None]).flatten(0, 1)
    return ogrid_sparse


def unique_indices(indices: Tensor, dim: int = 0) -> Tensor:
    """Compute the indices corresponding to the unique value.

    Args:
        indices: (N,K) Coordinate inputs.
        dim: Dimension to compute unique operation over.

    Returns:
        The indices corresponding to the selected values.
    """
    out: Tuple[Tensor, Tensor] = torch.unique(indices, return_inverse=True, dim=dim)
    unique, inverse = out
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    inv = inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
    inv, _ = inv.sort()
    return inv
