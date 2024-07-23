from typing import Tuple

import polars as pl
import torch
from kornia.geometry.conversions import euler_from_quaternion
from torch import Tensor


def polars_to_torch(
    annotations: pl.DataFrame, columns: Tuple[str, ...], device: torch.device
) -> Tensor:
    annotations_tch = torch.as_tensor(
        annotations.select(columns).to_numpy(), device=device
    )
    if annotations_tch.shape[0] == 0:
        return torch.empty((0, 10), device=device)

    _, _, yaws = euler_from_quaternion(*annotations_tch[:, 6:10].t())
    annotations_tch = torch.cat(
        [annotations_tch[:, :6], yaws[:, None], annotations_tch[:, 10:]], dim=-1
    )
    return annotations_tch
