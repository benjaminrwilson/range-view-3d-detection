"""Spatial kernels (i.e., filters).

Reference: https://en.wikipedia.org/wiki/Filter_(signal_processing)
"""

from typing import Tuple

import torch
from torch import Tensor

from torchbox3d.math.ops.index import ogrid_sparse_neighborhoods


@torch.jit.script
def gaussian_kernel(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Evaluate the univariate Gaussian kernel at x, parameterized by mu and sigma.

    f(x) = N(x; mu, sigma**2)

    Args:
        x: (N,) Gaussian support to be evaluated.
        mu: Mean parameter of the Gaussian.
        sigma: Width parameter of the Gaussian.

    Returns:
        The Gaussian kernel evaluated at x.
    """
    return torch.exp(-0.5 * (x - mu) ** 2 / sigma**2)


@torch.jit.script
def ogrid_sparse_gaussian(
    mus: Tensor, sigmas: Tensor, radius: int
) -> Tuple[Tensor, Tensor]:
    """Compute sparse Gaussian neighbhorhoods.

    Args:
        mus: (N,2) Centers of each sparse Gaussian.
        sigmas: (N,1) Width of each sparse Gaussian.
        radius: Maximum distance for non-zero response from
            the respective sparse Gaussian mean.

    Returns:
        The Gaussian kernel responses and the corresponding indice.
    """
    ogrid_offsets = ogrid_sparse_neighborhoods(mus, [radius, radius])
    mus = mus.repeat_interleave(int(radius**2), 0)
    sigmas = sigmas.repeat_interleave(int(radius**2), 0)
    response: Tensor = gaussian_kernel(ogrid_offsets, mus, sigmas).prod(
        dim=-1, keepdim=True
    )
    return response, ogrid_offsets
