"""Numpy-based conversions."""

import math

import numba as nb
import numpy as np


def build_range_view_coordinates(
    cart: np.array,
    sph: np.array,
    laser_numbers: np.array,
    laser_mapping: np.array,
    n_inclination_bins: int = 64,
    n_azimuth_bins: int = 1800,
) -> np.array:
    """Convert a set of points in R^3 (x,y,z) to range image of shape (n_inclination_bins,n_azimuth_bins,range).

    Args:
        cart: (N,3) Cartesian coordinates.
        sph: (N,3) Spherical coordinates.
        laser_numbers: (N,) Integer laser ids.
        laser_mapping: Mapping from laser number to row.
        n_inclination_bins: Number of inclination bins.
        n_azimuth_bins: Number of azimuthal bins.

    Returns:
        The range image containing range, intensity, and nanosecond offset.
    """
    azimuth = sph[..., 0]
    radius = sph[..., 2]

    azimuth += math.pi
    azimuth *= n_azimuth_bins / math.tau
    azimuth_index = (n_azimuth_bins - azimuth - 1).round()

    inclination_index = n_inclination_bins - laser_mapping[laser_numbers] - 1
    azimuth_index = np.clip(azimuth_index, 0, n_azimuth_bins - 1)
    hybrid = np.zeros_like(cart)
    hybrid[:, 0] = inclination_index
    hybrid[:, 1] = azimuth_index
    hybrid[:, 2] = radius
    return hybrid


def cart_to_sph(
    cart: np.ndarray,
) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        cart: (N,3) Cartesian coordinates.

    Returns:
        (N,3) Spherical coordinates: azimuth, inclination, and radius.
    """
    x = cart[..., 0]
    y = cart[..., 1]
    z = cart[..., 2]

    hypot_xy = np.hypot(x, y)
    radius = np.hypot(hypot_xy, z)
    inclination = np.arctan2(z, hypot_xy)
    azimuth = np.arctan2(y, x)

    sph = np.zeros_like(cart)
    sph[..., 0] = azimuth
    sph[..., 1] = inclination
    sph[..., 2] = radius
    return sph


def sph_to_cart(
    sph: np.ndarray,
) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    Reference:
        https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

    Args:
        cart: (N,3) Cartesian coordinates.

    Returns:
        (N,3) Spherical coordinates: azimuth, inclination, and radius.
    """
    azimuth = sph[..., 0]
    inclination = sph[..., 1]
    radius = sph[..., 2]

    rcos_theta = radius * np.cos(inclination)
    x = rcos_theta * np.cos(azimuth)
    y = rcos_theta * np.sin(azimuth)
    z = radius * np.sin(inclination)

    cart = np.zeros_like(sph)
    cart[..., 0] = x
    cart[..., 1] = y
    cart[..., 2] = z
    return cart


@nb.njit(nogil=True)  # type: ignore
def z_buffer(
    indices: np.ndarray,
    distances: np.ndarray,
    features: np.ndarray,
    height: int,
    width: int,
    min_distance: int = 1.0,
) -> np.ndarray:
    num_channels = features.shape[0]
    num_pixels = height * width
    image = np.zeros((num_channels, num_pixels), dtype=np.float32)
    buffer = np.full(num_pixels, np.inf, dtype=np.float32)

    row_indices, column_indices = indices
    raveled_indices = row_indices * width + column_indices
    for i, img_idx in enumerate(raveled_indices):
        if distances[i] < min_distance:
            continue
        if distances[i] < buffer[img_idx]:
            image[:, img_idx] = features[:, i]
            buffer[img_idx] = distances[i]
    return np.ascontiguousarray(image.reshape((num_channels, height, width)))
