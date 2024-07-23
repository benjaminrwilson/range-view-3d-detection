"""Argoverse dataset utilities."""

import logging
from pathlib import Path
from typing import Dict

import torch
from av2.utils.io import read_feather

from torchbox3d.datasets.argoverse.constants import LABEL_ATTR
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import Data

# Logging.
logger = logging.getLogger(__file__)


def read_sweep_data(
    dataset_dir: Path,
    split: str,
    log_id: str,
    sensor_name: str,
    timestamp_ns: int,
    category_to_index: Dict[str, int],
) -> Data:
    """Read sweep data into memory.

    Args:
        dataset_dir: Root dataset directory.
        split: Dataset split name.
        log_id: Log id.
        sensor_name: Sensor name.
        timestamp_ns: Nanosecond timestamp corresponding to the sweep.
        category_to_index: Mapping from the category name to the integer index.

    Returns:
        The sweep data.
    """
    log_dir = dataset_dir / split / log_id
    sweep_path = log_dir / "sensors" / sensor_name / f"{timestamp_ns}.feather"
    lidar = torch.as_tensor(
        read_feather(sweep_path).loc[:, ["x", "y", "z", "intensity"]].to_numpy(),
        dtype=torch.float,
    )
    lidar[..., 3] /= 255.0  # Normalize intensity values.

    cuboids = None
    if split != "test":
        annotations_path = log_dir / "annotations.feather"
        annotations = read_feather(annotations_path)

        timestamp_ns = int(sweep_path.stem)
        # We loaded all the annotations --- filter to the current sweep.
        annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]
        # Filter annotations with no points in them.
        annotations = annotations[annotations["num_interior_pts"] > 0]

        cuboid_params = torch.as_tensor(
            annotations.loc[:, list(LABEL_ATTR)].to_numpy(),
            dtype=torch.float,
        )

        scores = torch.ones(len(cuboid_params), dtype=torch.float)
        cuboids = Cuboids(params=cuboid_params, scores=scores)

    log_id = sweep_path.parent.parent.parent.stem
    uuid = (log_id, str(timestamp_ns))

    coordinates_m = lidar[..., :3]
    values = lidar[..., 3:]
    counts = torch.ones_like(values)
    datum = Data(
        coordinates_m=coordinates_m,
        values=values,
        counts=counts,
        cuboids=cuboids,
        uuids=uuid,
    )
    return datum
