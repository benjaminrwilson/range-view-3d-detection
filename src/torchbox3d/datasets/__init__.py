"""Package containing all datasets."""

import math
from typing import List

from av2.evaluation import SensorCompetitionCategories
from av2.evaluation.detection.eval import DetectionCfg

from torchbox3d.datasets.argoverse.constants import (
    NuscenesCompetitionCategories,
    WaymoCompetitionCategories,
)


def detection_cfg_factory(
    dataset_dir, dataset_name: str, valid_categories: List[str]
) -> DetectionCfg:
    """Get the 3D detection configuration for a specific dataset.

    Args:
        dataset_dir: Dataset root directory.
        dataset_name: 3D detection dataset name.

    Returns:
        Detection configuration.
    """
    if dataset_name.upper() == "AV2":
        categories = SensorCompetitionCategories
        eval_only_roi_instances = True
        max_range_m = 150.0
    elif dataset_name.upper() == "WAYMO":
        categories = WaymoCompetitionCategories
        eval_only_roi_instances = False
        max_range_m = math.inf
    # elif dataset_name.upper() == "NUSCENES":
    elif "NUSCENES" in dataset_name.upper():
        categories = NuscenesCompetitionCategories
        eval_only_roi_instances = False
        max_range_m = 55.0

    categories = set(x.value for x in categories) & set(valid_categories)
    return DetectionCfg(
        dataset_dir=dataset_dir,
        categories=tuple(sorted(categories)),
        max_range_m=max_range_m,
        eval_only_roi_instances=eval_only_roi_instances,
    )
