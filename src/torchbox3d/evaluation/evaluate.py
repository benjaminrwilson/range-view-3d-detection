"""Evaluation tool for the Waymo Open dataset."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import logging
import math
from pathlib import Path
from typing import Final, Tuple, cast

import numpy as np
import polars as pl
import wandb
from numpy.typing import NDArray
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import wod_detection_evaluator
from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2
from waymo_open_dataset.protos.metrics_pb2 import Config

logger = logging.getLogger(__name__)

DTS_COLS: Final = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "object_type",
    "score",
]
GTS_COLS: Final = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "object_type",
    "difficulty_level",
]

###############################################
# BEGIN CONSTUCTING WAYMO EVALUATION COLUMNS. #
###############################################

CATEGORY_TO_INDEX: Final = {"VEHICLE": 1, "PEDESTRIAN": 2, "SIGN": 3, "CYCLIST": 4}

METRIC_NAMES: Final = [
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
    "AP",
]

LEVELS: Final = (
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
    1,
    2,
)

CATEGORIES: Final = (
    "VEHICLE",
    "VEHICLE",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "SIGN",
    "SIGN",
    "CYCLIST",
    "CYCLIST",
    "VEHICLE",
    "VEHICLE",
    "VEHICLE",
    "VEHICLE",
    "VEHICLE",
    "VEHICLE",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "PEDESTRIAN",
    "SIGN",
    "SIGN",
    "SIGN",
    "SIGN",
    "SIGN",
    "SIGN",
    "CYCLIST",
    "CYCLIST",
    "CYCLIST",
    "CYCLIST",
    "CYCLIST",
    "CYCLIST",
)

LOWER_RANGE: Final = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    30.0,
    30.0,
    50.0,
    50.0,
    0.0,
    0.0,
    30.0,
    30.0,
    50.0,
    50.0,
    0.0,
    0.0,
    30.0,
    30.0,
    50.0,
    50.0,
    0.0,
    0.0,
    30.0,
    30.0,
    50.0,
    50.0,
)

UPPER_RANGE: Final = (
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    30.0,
    30.0,
    50.0,
    50.0,
    math.inf,
    math.inf,
    30.0,
    30.0,
    50.0,
    50.0,
    math.inf,
    math.inf,
    30.0,
    30.0,
    50.0,
    50.0,
    math.inf,
    math.inf,
    30.0,
    30.0,
    50.0,
    50.0,
    math.inf,
    math.inf,
)

###############################################
# END CONSTUCTING WAYMO EVALUATION COLUMNS.   #
###############################################


def pull_wandb_feather(project_name: str, entity: str, tag: str) -> Path:
    """Load an artifact from Wandb.

    Args:
        model_name: Wandb model name.
        project_name: WandB project name.
        entity: WandB entity name.
        tag: WandB tag.

    Returns:
        Artifact path.
    """
    path = f"{entity}/{project_name}/val_results:{tag}"
    run = wandb.init()
    artifact = run.use_artifact(path, type="metrics")
    artifact.download()
    return Path(f"artifacts/val_results:{tag}")


def quat_to_yaw(quat_wxyz: np.array) -> np.array:
    """Convert a scalar first quaternion to yaw (rotation about the vertical).

    Args:
        quat_wxyz: (N,4) Scalar-first quaternion.

    Returns:
        (N,) Yaw in radians.
    """
    (qw, qx, qy, qz) = (
        quat_wxyz[:, 0],
        quat_wxyz[:, 1],
        quat_wxyz[:, 2],
        quat_wxyz[:, 3],
    )
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return cast(NDArray, np.arctan2(siny_cosp, cosy_cosp))


def build_config(true_positive_type: str) -> Config:
    """Build a custom detection evaluation config."""
    config = metrics_pb2.Config()

    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
    difficulty = config.difficulties.add()
    difficulty.levels.append(label_pb2.Label.LEVEL_1)
    difficulty.levels.append(label_pb2.Label.LEVEL_2)
    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.RANGE)
    difficulty = config.difficulties.add()
    difficulty.levels.append(label_pb2.Label.LEVEL_1)
    difficulty.levels.append(label_pb2.Label.LEVEL_2)

    config.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
    config.iou_thresholds.append(0.0)
    config.iou_thresholds.append(0.7)
    config.iou_thresholds.append(0.5)
    config.iou_thresholds.append(0.5)
    config.iou_thresholds.append(0.5)

    if true_positive_type == "BEV":
        config.box_type = label_pb2.Label.Box.TYPE_2D
    elif true_positive_type == "3D":
        config.box_type = label_pb2.Label.Box.TYPE_3D
    else:
        raise NotImplementedError("Not implemented.")

    for i in range(100):
        config.score_cutoffs.append(i * 0.01)
    config.score_cutoffs.append(1.0)
    return config


def prepare_data(
    dts: pl.DataFrame, gts: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    gts = gts.filter(pl.col("num_interior_pts") > 0)
    point_mask = pl.col("num_interior_pts") <= 5
    no_difficulty_mask = pl.col("difficulty_level") == 0

    level_2 = (point_mask & no_difficulty_mask) * 2
    level_1 = (~point_mask & no_difficulty_mask) * 1

    expr = pl.max_horizontal(pl.col("difficulty_level"), level_1 + level_2)
    gts = gts.with_columns(difficulty_level=expr)

    assert (
        gts["difficulty_level"]
        .value_counts()
        .select((pl.col("difficulty_level") == 0).any().not_())
        .item()
    )

    gts = gts.with_columns(
        category=pl.col("category").map_dict(CATEGORY_TO_INDEX)
    ).rename({"category": "object_type"})
    dts = dts.with_columns(
        category=pl.col("category").map_dict(CATEGORY_TO_INDEX)
    ).rename({"category": "object_type"})
    return dts, gts


def evaluate() -> None:
    """Evaluate the Waymo Open dataset."""
    project_name = "cvpr_waymo"
    entity = "benjaminrwilson"
    version = "2104"

    root_dir = pull_wandb_feather(project_name, entity, f"v{version}")
    pds_path = root_dir / "detections.feather"
    gts_path = root_dir / "annotations.feather"

    dts = pl.read_ipc(pds_path)
    gts = pl.read_ipc(gts_path)
    results = evaluate_waymo(dts, gts)
    print(results)


def evaluate_waymo(dts: pl.DataFrame, gts: pl.DataFrame) -> pl.DataFrame:
    dts, gts = prepare_data(dts, gts)
    uuid_to_dts = {
        key: group.select(pl.col(DTS_COLS)).to_numpy()
        for key, group in dts.group_by(["log_id", "timestamp_ns"])
    }
    uuid_to_gts = {
        key: group.select(pl.col(GTS_COLS)).to_numpy()
        for key, group in gts.group_by(["log_id", "timestamp_ns"])
    }

    dts_list = []
    gts_list = []
    pd_frame_id_list = []
    gt_frameid_list = []
    for i, (k, v) in enumerate(uuid_to_gts.items()):
        if k in uuid_to_dts:
            pd_frame_id_list.append(np.full(len(uuid_to_dts[k]), fill_value=i))
            dts_list.append(uuid_to_dts[k])
        else:
            logger.info("No detections found in uuid: %s.", k)
        gt_frameid_list.append(np.full(len(v), fill_value=i))
        gts_list.append(v)

    gts_boxes = np.concatenate(gts_list)
    dts_boxes = np.concatenate(dts_list)

    dts_frame_ids = np.concatenate(pd_frame_id_list)
    gts_frame_ids = np.concatenate(gt_frameid_list)

    gts_yaws = quat_to_yaw(gts_boxes[:, 6:10])[:, None]
    dts_yaws = quat_to_yaw(dts_boxes[:, 6:10])[:, None]

    dts_scores = dts_boxes[:, -1].astype(np.float32)
    dts_overlap_nlz = np.zeros_like(dts_scores, dtype=bool)

    dts_types = dts_boxes[:, -2].astype(np.uint8)
    gts_types = gts_boxes[:, -2].astype(np.uint8)
    difficulty_level = gts_boxes[:, -1].astype(np.uint8)

    dts_boxes = np.concatenate([dts_boxes[:, :6], dts_yaws], axis=1).astype(np.float32)
    gts_boxes = np.concatenate([gts_boxes[:, :6], gts_yaws], axis=1).astype(np.float32)

    dts_dict = {
        "prediction_frame_id": dts_frame_ids,
        "prediction_bbox": dts_boxes,
        "prediction_type": dts_types,
        "prediction_score": dts_scores,
        "prediction_overlap_nlz": dts_overlap_nlz,
    }

    gts_dict = {
        "ground_truth_frame_id": gts_frame_ids,
        "ground_truth_bbox": gts_boxes,
        "ground_truth_type": gts_types,
        "ground_truth_difficulty": difficulty_level,
    }

    # Compute BEV metrics.
    config = build_config("BEV")
    evaluator = wod_detection_evaluator.WODDetectionEvaluator(config)
    evaluator.update_state(gts_dict, dts_dict)
    ap_bev, aph, apl, pr, prh, prl, breakdown = evaluator.result()
    ap_bev = ap_bev.numpy()

    # Compute 3D metrics.
    config = build_config("3D")
    evaluator = wod_detection_evaluator.WODDetectionEvaluator(config)
    evaluator.update_state(gts_dict, dts_dict)
    ap_3d, aph, apl, pr, prh, prl, breakdown = evaluator.result()
    ap_3d = ap_3d.numpy()

    tp_type = ["BEV"] * len(UPPER_RANGE)
    results_bev = pl.DataFrame(
        {
            "metric_name": METRIC_NAMES,
            "type": tp_type,
            "category": CATEGORIES,
            "level": LEVELS,
            "r_lower": LOWER_RANGE,
            "r_upper": UPPER_RANGE,
            "value": ap_bev,
        }
    )

    tp_type = ["3D"] * len(UPPER_RANGE)
    results_3d = pl.DataFrame(
        {
            "metric_name": METRIC_NAMES,
            "type": tp_type,
            "category": CATEGORIES,
            "level": LEVELS,
            "r_lower": LOWER_RANGE,
            "r_upper": UPPER_RANGE,
            "value": ap_3d,
        }
    )

    results = pl.concat([results_bev, results_3d])
    return results


if __name__ == "__main__":
    evaluate()
