"""Coding and decoding."""

from typing import Final, Tuple, TypeVar

import polars as pl
import torch
from torch import Tensor

T = TypeVar("T")

SCHEMA: Final = {
    "tx_m": pl.Float32,
    "ty_m": pl.Float32,
    "tz_m": pl.Float32,
    "length_m": pl.Float32,
    "width_m": pl.Float32,
    "height_m": pl.Float32,
    "qw": pl.Float32,
    "qx": pl.Float32,
    "qy": pl.Float32,
    "qz": pl.Float32,
    "num_interior_pts": pl.UInt32,
    "score": pl.Float32,
    "category_index": pl.Int32,
    "batch_index": pl.Int32,
    "log_id": pl.Utf8,
    "timestamp_ns": pl.Int64,
}


def build_dataframe(
    params: Tensor,
    scores: Tensor,
    categories: Tensor,
    batch_index: Tensor,
    uuids: pl.DataFrame,
    idx_to_category: pl.DataFrame,
) -> pl.DataFrame:
    uuids = uuids.with_columns(timestamp_ns=pl.col("timestamp_ns").cast(pl.Int64))
    columns: Tuple[Tensor, ...] = params.split(1, dim=-1)
    tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz = tuple(
        col.flatten() for col in columns
    )
    dts = pl.DataFrame(
        {
            "tx_m": tx_m.tolist(),
            "ty_m": ty_m.tolist(),
            "tz_m": tz_m.tolist(),
            "length_m": length_m.tolist(),
            "width_m": width_m.tolist(),
            "height_m": height_m.tolist(),
            "qw": qw.tolist(),
            "qx": qx.tolist(),
            "qy": qy.tolist(),
            "qz": qz.tolist(),
            "score": scores.flatten().tolist(),
            "category_index": categories.int().flatten().tolist(),
            "batch_index": batch_index.int().flatten().tolist(),
        },
        schema_overrides=SCHEMA,
    ).lazy()

    task_frame_lazy = (
        idx_to_category.with_row_count("category_index")
        .cast({"category_index": pl.Int32})
        .lazy()
    )

    uuids_lazy = uuids.lazy()

    dts_lazy = (
        dts.join(uuids_lazy, on="batch_index")
        .join(task_frame_lazy, on="category_index")
        .drop(["category_index", "task_index", "task_offset"])
    )
    return dts_lazy.collect()


@torch.jit.script
def egovehicle_from_azimuth(
    xyz: Tensor, offset: Tensor, yaw: Tensor
) -> Tuple[Tensor, Tensor]:
    """Decode azimuth invariant coordinates into ego-vehicle coordinates.

    Args:
        xyz: (N,3) Tensor of points.
        offset: (N,3) Tensor of offsets.
        yaw: (N,) Signed rotation to the first principal axis.

    Returns:
        Offset and yaw tensors.
    """
    azimuth = torch.atan2(
        xyz[:, 1],
        xyz[:, 0],
    )

    sin = azimuth.sin()
    cos = azimuth.cos()

    x = cos * offset[:, 0] - sin * offset[:, 1]
    y = sin * offset[:, 0] + cos * offset[:, 1]
    z = offset[:, 2]

    offset = torch.stack([x, y, z], dim=1)
    yaw += azimuth[:, None]
    return offset, yaw


@torch.jit.script
def decode_range_view(
    regressands: Tensor,
    cart: Tensor,
    enable_azimuth_invariant_targets: bool,
) -> Tensor:
    """Decode the length, width, height cuboid parameterization.

    Args:
        regressands: (B,R,H,W) Regression output from the network.
        logits: (B,C,H,W) Logit output from the network.
        grid_data: Preprocessed input data.

    Returns:
        The decoded predictions.
    """
    dtype = regressands.dtype
    regressands = regressands.double()
    cart = cart.double()

    # Split up the regressands.
    offset = regressands[:, :3]
    lwh = regressands[:, 3:6].exp()

    sin = regressands[:, 6:7]
    cos = regressands[:, 7:8]

    ctrs = offset
    yaw = torch.atan2(sin, cos)
    if enable_azimuth_invariant_targets:
        offset, yaw = egovehicle_from_azimuth(cart.type_as(offset), offset, yaw)

    ctrs = cart + offset
    params = torch.cat((ctrs, lwh, yaw), dim=1)
    return params.type(dtype)
