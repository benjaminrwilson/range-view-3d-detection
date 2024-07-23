from typing import Optional

import numpy as np
import polars as pl
from polars import col

from torchbox3d.math.numpy.conversions import (
    build_range_view_coordinates,
    cart_to_sph,
    z_buffer,
)


def build_range_view(
    sweep: pl.DataFrame,
    laser_mapping: np.array,
    lidar_offset: np.array,
    timestamp_ns: Optional[int] = None,
    max_timestamp_ns: Optional[int] = None,
    num_lasers: int = 64,
    width: int = 1800,
):
    sweep = sweep.filter(
        col("laser_number").lt(num_lasers)
        # & col("offset_ns").le(max_timestamp_ns - timestamp_ns)
    )
    xyz_object = sweep.select(["x", "y", "z"]).to_numpy()
    # sweep = unmotion_compensate(sweep, poses, timestamp_ns, slerp)
    cart = sweep.select(["x", "y", "z"]).to_numpy() - lidar_offset
    sph = cart_to_sph(cart)
    laser_numbers = sweep["laser_number"].to_numpy().copy()
    intensity = sweep.select(["intensity"]).to_numpy()
    features = np.concatenate([sph, xyz_object, intensity], axis=1).transpose(1, 0)
    hybrid = build_range_view_coordinates(
        cart,
        sph,
        laser_numbers,
        laser_mapping,
        n_inclination_bins=num_lasers,
    )
    indices = np.ascontiguousarray(hybrid[:, :2].transpose(1, 0).astype(int))
    distances = hybrid[:, 2]
    buffer = z_buffer(indices, distances, features, height=num_lasers, width=width)
    return buffer
