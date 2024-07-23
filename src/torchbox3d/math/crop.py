"""Spatial cropping methods."""

from typing import List, Tuple

import torch
from torch import Tensor


@torch.jit.script
def crop_coordinates(
    coordinates_m: Tensor,
    lower_bound_inclusive: List[float],
    upper_bound_exclusive: List[float],
) -> Tuple[Tensor, Tensor]:
    """Crop an N-dimensional tensor.

    Args:
        coordinates_m: (N,D) Tensor of coordinates.
        lower_bound_inclusive: (D,) Minimum coordinate thresholds for cropping.
        upper_bound_exclusive: (D,) Maximum coordinate thresholds for cropping.

    Returns:
        (N,) Valid point mask.
    """
    D = min(
        coordinates_m.shape[-1],
        len(lower_bound_inclusive),
        len(upper_bound_exclusive),
    )
    lower_bound_inclusive = lower_bound_inclusive[:D]
    upper_bound_exclusive = upper_bound_exclusive[:D]

    lower = torch.as_tensor(lower_bound_inclusive, device=coordinates_m.device)
    upper = torch.as_tensor(upper_bound_exclusive, device=coordinates_m.device)
    mask = torch.logical_and(
        coordinates_m[..., :D] >= lower, coordinates_m[..., :D] < upper
    ).all(dim=1)
    return coordinates_m[mask], mask
