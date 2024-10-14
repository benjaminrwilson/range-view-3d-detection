"""3D detection model."""

from __future__ import annotations

import logging
import uuid
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, DefaultDict, Dict, Final, List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import wandb
from av2.evaluation.detection.eval import UUID_COLUMNS, DetectionCfg, evaluate
from filelock import FileLock
from kornia.geometry.conversions import (
    quaternion_from_euler,
    quaternion_to_rotation_matrix,
)
from omegaconf import MISSING, DictConfig
from polars import col, lit
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from torchbox3d.datasets import detection_cfg_factory
from torchbox3d.evaluation.evaluate import evaluate_waymo
from torchbox3d.math.ops.coding import build_dataframe
from torchbox3d.nn.meta.arch import MetaDetector
from torchbox3d.rendering.tensorboard import to_logger
from torchbox3d.utils.hydra import flatten

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(100)

logger = logging.getLogger(__name__)

SERIALIZED_SCHEMA: Final = (
    ("tx_m", np.float32),
    ("ty_m", np.float32),
    ("tz_m", np.float32),
    ("length_m", np.float32),
    ("width_m", np.float32),
    ("height_m", np.float32),
    ("qw", np.float32),
    ("qx", np.float32),
    ("qy", np.float32),
    ("qz", np.float32),
    ("score", np.float32),
    ("log_id", np.object_),
    ("timestamp_ns", np.uint64),
    ("category_index", np.int32),
)

CUBOID_COLUMN_NAMES: Final = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "task_id",
    "offset",
    "batch_index",
)

CATEGORY_SCHEMA: Final = (
    ("task_id", pl.Int32),
    ("offset", pl.Int32),
    ("category", pl.Utf8),
)


@dataclass(unsafe_hash=True)
class Detector(MetaDetector):
    """3D Object Detection Model.

    Args:
        batch_size: Batch size used to determine training learning rate.
        dataset_name: Training / validation / test dataset name.
        debug: Boolean flag to enable debugging.
        dst_dir: Output directory for any artifacts (e.g., predictions).
        root_dir: Root directory for loading data.
        tasks: Groups of classes for training.
        train_log_freq: Training logging frequency.
        val_log_freq: Validation logging frequency.
        target_encoding_config: Target encoding configuration for training.
    """

    batch_size: int = MISSING
    dataset_name: str = MISSING
    evaluation_split_name: str = MISSING
    debug: bool = MISSING
    dst_dir: str = MISSING
    precision: str = MISSING
    eval_precision: str = MISSING
    model_name: str = MISSING
    num_devices: int = MISSING
    num_workers: int = MISSING
    root_dir: str = MISSING
    tasks: DictConfig = MISSING
    train_log_freq: int = MISSING
    val_log_freq: int = MISSING
    enable_database: bool = MISSING

    post_processing_config: DictConfig = MISSING

    augmentations_config: Optional[DictConfig] = None
    db_config: Optional[DictConfig] = None

    _trainer: DictConfig = MISSING
    uuid: str = str(uuid.uuid4())

    def __post_init__(self) -> None:
        """Initialize network."""
        super().__post_init__()
        rank_zero_info("Initializing Detector ...")

        def _build_offset_mapping(tasks: DictConfig) -> pl.DataFrame:
            task_data = defaultdict(list)
            for k, task_categories in tasks.items():
                for offset, category in enumerate(sorted(task_categories)):
                    task_data["task_id"].append(k)
                    task_data["offset"].append(offset)
                    task_data["category"].append(category)

            task_frame = pl.DataFrame(task_data, schema=CATEGORY_SCHEMA)
            return task_frame

        self.category_id_to_category = _build_offset_mapping(self.tasks)
        self.category_priors = self.compute_category_priors()
        self.save_hyperparameters()
        self.save_hyperparameters(
            flatten(self._scheduler, parent_key="scheduler"),
        )
        self.save_hyperparameters(
            flatten(self._optimizer, parent_key="optimizer"),
        )
        self.save_hyperparameters(
            flatten(self._backbone, parent_key="neck"),
        )
        self.save_hyperparameters(
            flatten(self._head, parent_key="head"),
        )
        self.save_hyperparameters(
            flatten(self._decoder, parent_key="decoder"),
        )

    def compute_category_priors(self) -> Tensor:
        split_dir = Path(self.root_dir) / "train"
        annotations_list: List[pl.DataFrame] = []
        for log_path in sorted(split_dir.glob("*")):
            annotations_path = log_path / "annotations.feather"
            annotations = pl.read_ipc(annotations_path)
            annotations = annotations.with_columns(
                pl.Series("log_id", [log_path.stem] * len(annotations))
            )

            if len(annotations) > 0:
                annotations_list.append(annotations)

        device = self.device
        counts = torch.zeros(
            len(self.category_id_to_category),
            device=device,
            dtype=torch.float32,
        )
        if len(annotations_list) > 0:
            annotations = pl.concat(annotations_list).filter(
                pl.col("num_interior_pts") > 0
            )

            value_counts = annotations["category"].value_counts()
            value_counts = self.category_id_to_category.join(
                value_counts, on="category", how="left"
            )
            counts = torch.as_tensor(
                value_counts["count"].cast(pl.Int64).to_numpy().copy(),
                device=device,
                dtype=torch.float32,
            )

        return counts

    def forward(
        self, input: Dict[str, Tensor], return_loss: bool = True
    ) -> Dict[str, Any]:
        """Compute CenterPoint forward pass."""
        backbone = cast(Dict[str, Tensor], self.backbone.forward(input))
        input["category_priors"] = self.category_priors
        head, losses = self.head.forward(
            backbone,
            input,
            return_loss=return_loss,
        )
        return {
            "backbone": backbone,
            "head": head,
        } | losses

    def on_train_start(self) -> None:
        """Initialize Tensorboard hyperparameters if applicable."""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if dist.get_rank() == 0:
                objects = [self.uuid for _ in range(world_size)]
            else:
                objects = [None for _ in range(world_size)]
            dist.broadcast_object_list(objects)
            self.uuid = objects[0]
            logger.info("UUID: %s.\n", self.uuid)
        if self.logger is None:
            return
        if not isinstance(self.logger, TensorBoardLogger):
            return
        self.logger.log_hyperparams(
            Namespace(kwargs=self.hparams),
            {
                "hp/CDS": 0,
                "hp/AP": 0,
                "hp/ATE": 0,
                "hp/ASE": 0,
                "hp/AOE": 0,
            },
        )

    def training_step(self, data: Dict[str, Tensor], idx: int) -> Dict[str, Tensor]:
        """Training step."""
        outputs = self.forward(data)
        losses = {k: v for k, v in outputs.items() if isinstance(v, Tensor)}
        self.log_dict(
            losses,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return outputs

    def on_train_batch_end(  # type: ignore[override]
        self,
        outputs: Dict[str, Tensor],
        data: Dict[str, Union[pl.DataFrame, Tensor]],
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Log visualizations at the end of the training batch."""
        if self.global_step % self.train_log_freq == 0 and self.local_rank == 0:
            batch_index = 0
            uuids = cast(pl.DataFrame, data["uuids"])

            # Grab the first batch index to prevent decoding all during training.
            for stride, multiscale_outputs in outputs["head"].items():
                for task_id, task_outputs in multiscale_outputs.items():
                    if isinstance(task_id, int):
                        for k, v in task_outputs.items():
                            outputs["head"][stride][task_id][k] = v[
                                batch_index : batch_index + 1
                            ]
                    else:
                        outputs["head"][stride][task_id] = task_outputs[
                            batch_index : batch_index + 1
                        ]

            for k, v in data.items():
                if isinstance(v, pl.DataFrame):
                    data[k] = v.filter(
                        pl.col("batch_index") == batch_index
                    ).with_columns(batch_index=0)
                else:
                    data[k] = v[batch_index : batch_index + 1]

            params, scores, categories, batch_indices = self.decoder.decode(
                outputs["head"],
                post_processing_config=self.post_processing_config,
                task_config=self.tasks,
            )

            dts = build_dataframe(
                params=params,
                scores=scores,
                categories=categories,
                batch_index=batch_indices,
                uuids=uuids[batch_index : batch_index + 1].with_columns(batch_index=0),
                idx_to_category=self.category_id_to_category,
            )

            to_logger(
                dts,
                uuids,
                data,
                batch_index=0,
                network_outputs=outputs,
                tasks_cfg=self.tasks,
                trainer=self.trainer,
                debug=self.debug,
            )

        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except:
            pass

    @torch.no_grad()
    def validation_step(
        self, data: Dict[str, Tensor], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Take a validation step.

        Args:
            data: Input data.
            batch_idx: Batch index.

        Returns:
            The validation outputs.
        """
        device_type = str(self.device.type)
        dtype = (
            "32" if str(self.eval_precision) not in ("16", "bf16") else torch.float16
        )
        with torch.autocast(device_type=device_type, dtype=dtype):
            outputs = self.forward(data, return_loss=True)
            uuids = data["uuids"]
            params, scores, categories, batch_index = self.decoder.decode(
                outputs["head"],
                post_processing_config=self.post_processing_config,
                task_config=self.tasks,
            )

            dts = build_dataframe(
                params,
                scores,
                categories,
                batch_index,
                uuids,
                self.category_id_to_category,
            )
            if (
                self.debug
                or batch_idx % self.val_log_freq == 0
                and self.local_rank == 0
            ):
                to_logger(
                    dts,
                    uuids,
                    data,
                    batch_index=0,
                    network_outputs=outputs,
                    tasks_cfg=self.tasks,
                    trainer=self.trainer,
                    debug=self.debug,
                )

            for (log_id, timestamp_ns), group in dts.group_by(
                ["log_id", "timestamp_ns"],
                maintain_order=True,
            ):
                dst = (
                    Path(self.dst_dir)
                    / "predictions"
                    / self.uuid
                    / log_id
                    / f"{timestamp_ns}.feather"
                )

                with FileLock(str(dst) + ".lock"):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    group.write_ipc(dst)

            losses = {
                f"val/{k}": v for k, v in outputs.items() if isinstance(v, Tensor)
            }
            self.log_dict(
                losses,
                batch_size=self.batch_size,
                sync_dist=True,
            )
        return outputs

    @torch.no_grad()
    def test_step(
        self, data: Dict[str, Tensor], batch_idx: int
    ) -> Optional[STEP_OUTPUT]:
        """Compute the network predictions.

        Args:
            data: (T,) List of network task outputs.
            batch_idx: Batch index.

        Returns:
            The network predictions.
        """
        return None

    def on_validation_end(self) -> None:
        """Run validation end."""
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return
        # if self.current_epoch != self.trainer.max_epochs - 1:
        #     return

        # Ensure we wait for all predictions to be written.
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except:
            pass

        logger.info("Starting on_evaluation_end ...")

        # Only evaluate on one device.
        if self.local_rank == 0:
            pattern = f"predictions/{self.uuid}/*/*.feather"
            src = Path(self.dst_dir) / pattern
            paths = sorted(Path(self.dst_dir).glob(pattern))
            if len(paths) == 0:
                logger.info("No detections found. Skipping validation ...")
                return
            dts = pl.scan_ipc(src, memory_map=True).collect()
            if len(dts) == 0:
                logger.info("No detections found. Skipping validation ...")
                return

            valid_categories = list(chain.from_iterable(self.tasks.values()))

            ##############################################################
            # Begin processing detections and ground truth annotations.  #
            ##############################################################

            dataset_name = self.dataset_name
            split = self.evaluation_split_name
            dataset_dir = Path(self.root_dir) / split

            metadata = pl.DataFrame(
                [
                    pl.Series("log_id", [path.parent.stem for path in paths]),
                    pl.Series("timestamp_ns", [int(path.stem) for path in paths]),
                ]
            )

            # metadata = cast(pl.DataFrame, self.trainer.datamodule.val_dataset.metadata)

            dts, gts, cfg = prepare_for_evaluation(
                dts,
                dataset_name,
                split,
                dataset_dir,
                valid_categories,
                metadata,
            )

            if len(gts) == 0:
                logger.info("No ground truth found. Skipping validation ...")
                return

            logger.info("Starting evaluation ...")
            if self.dataset_name == "av2":
                _, _, metrics = evaluate(dts.to_pandas(), gts.to_pandas(), cfg)

                metrics["AP"] *= 100
                metrics = pl.from_pandas(metrics, include_index=True).rename(
                    {"None": "category"}
                )

                metrics = format_evaluation_metrics(metrics, dts, gts)
                for (
                    category,
                    ap,
                    ate,
                    ase,
                    aoe,
                    cds,
                ) in metrics.select(
                    pl.col(["category", "AP", "ATE", "ASE", "AOE", "CDS"])
                ).rows():
                    if self.logger is not None:
                        name = f"AP/{category}"
                        self.logger.experiment.log({name: ap})

                        name = f"CDS/{category}"
                        self.logger.experiment.log({name: cds})
                print(metrics)

            elif self.dataset_name == "waymo":
                metrics = evaluate_waymo(dts, gts)
                metrics = metrics.with_columns((pl.col("value") * 100.0).round(3))
                expr = (pl.col("level") == 1) & pl.col("category").is_in(cfg.categories)
                metrics = metrics.filter(expr)
                metrics = metrics.sort(by=("category"))
                print(metrics)
                for (
                    metric_name,
                    metric_type,
                    category,
                    level,
                    lower_interval,
                    upper_interval,
                    value,
                ) in metrics.rows():
                    name = f"{metric_name}-{metric_type}-L{level}/{category}/{lower_interval}-{upper_interval}"
                    if self.logger is not None:
                        self.logger.experiment.log({name: value})

            # Log artifacts.
            metrics_dir = Path(self.dst_dir) / "results" / self.uuid
            metrics_dir.mkdir(parents=True, exist_ok=True)
            dts.write_ipc(
                metrics_dir / "detections.feather",
            )
            gts.write_ipc(
                metrics_dir / "annotations.feather",
            )
            metrics.write_ipc(
                metrics_dir / "metrics.feather",
                compression="uncompressed",
            )
            artifact = wandb.Artifact(name=f"{split}_results", type="metrics")
            artifact.add_file(str(metrics_dir / "detections.feather"))
            artifact.add_file(str(metrics_dir / "annotations.feather"))
            artifact.add_file(str(metrics_dir / "metrics.feather"))
            wandb.log_artifact(artifact)

        # Wait for all gpus.
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
        except:
            pass


def prepare_for_evaluation(
    pds: pl.DataFrame,
    dataset_name: str,
    split: str,
    dataset_dir: Path,
    valid_categories: List[str],
    metadata: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame, DetectionCfg]:
    """Prepare the predictions and ground truth annotations for evaluation.

    Args:
        pds: DataFrame of predictions.
        dataset_name: Dataset name (e.g., "av2").
        split: Split name (e.g., "val").
        dataset_dir: Dataset directory.

    Returns:
        Predictions and ground truths.
    """
    cfg = detection_cfg_factory(dataset_dir, dataset_name, valid_categories)
    logger.info("Using the following detection configuration: %s", cfg)

    ##############################################################
    # Begin processing detections and ground truth annotations.  #
    ##############################################################

    norms = np.linalg.norm(
        pds.lazy().select(col(["tx_m", "ty_m", "tz_m"])).collect().to_numpy(),
        axis=-1,
    )

    pds_lazy = (
        pds.with_columns(pl.Series("norm", norms))
        .lazy()
        .filter(col("norm").le(lit(cfg.max_range_m)))
        .sort(col("score"), descending=True)
        .unique()
    )

    annotation_paths = sorted(dataset_dir.glob("*/annotations.feather"))
    logger.info(f"Evaluating on the following splits: {split}.")
    logger.info("Loading validation data ...")

    def _read_frame(path: Path) -> pl.DataFrame:
        datum = pl.read_ipc(path, memory_map=False)
        log_ids = pl.Series("log_id", [path.parent.stem] * len(datum))
        return datum.with_columns(log_ids)

    # Load all ground truth annotations.
    # Filter based off of valid categories and valid uuids.
    gts_lazy = (
        pl.concat([_read_frame(path) for path in annotation_paths])
        .lazy()
        .cast({"timestamp_ns": pl.Int64})
        .sort(pl.col(UUID_COLUMNS + ("category",)))
        .filter(pl.col("category").is_in(valid_categories))
        .join(
            metadata.lazy().cast({"timestamp_ns": pl.Int64}),
            on=("log_id", "timestamp_ns"),
        )
        .cast({"timestamp_ns": pl.Int64})
    )

    valid_uuids = gts_lazy.select(col(UUID_COLUMNS)).unique()
    num_unique_uuids = valid_uuids.collect().shape[0]
    logger.info(f"Using {num_unique_uuids} uuids for evaluation ...")

    pds = pds_lazy.join(valid_uuids, on=UUID_COLUMNS).collect()
    gts = gts_lazy.join(valid_uuids, on=UUID_COLUMNS).collect()
    return pds, gts, cfg


# def compute_amodal_weights(
#     batch_annotations_i: Tensor, strided_mask: Tensor, strided_points: Tensor
# ) -> Tensor:
#     yaws = batch_annotations_i[:, 6]
#     pitch = torch.zeros_like(yaws)
#     roll = torch.zeros_like(yaws)
#     quat = torch.stack(quaternion_from_euler(roll, pitch, yaws), dim=-1)
#     rots = quaternion_to_rotation_matrix(quat)
#     amodal_weights_list = []
#     for j, m in enumerate(strided_mask):
#         mask_points = strided_points[m]
#         if len(mask_points) == 0:
#             amodal_weights_list.append(0.0)
#             continue

#         points_obj = (rots[j].t() @ (mask_points - batch_annotations_i[j, :3]).t()).t()

#         mins, _ = points_obj.min(dim=0)
#         maxes, _ = points_obj.max(dim=0)

#         # amodal_weight = 1.0 - (
#         #     (maxes - mins).clamp(0).prod() / batch_annotations[i][j, 3:6].prod()
#         # ).clamp(0.0, 1.0)
#         amodal_weight = 1.0 - (
#             (maxes[2] - mins[2]).clamp(0).prod() / batch_annotations_i[j, 5].prod()
#         ).clamp(0.0, 1.0)
#         amodal_weights_list.append(amodal_weight)

#     amodal_weights = strided_points.new(amodal_weights_list)
#     return amodal_weights


def format_evaluation_metrics(
    metrics: pl.DataFrame,
    dts: pl.DataFrame,
    gts: pl.DataFrame,
    num_digits: int = 2,
) -> pl.DataFrame:
    """Format evaluation metrics for logging."""
    metrics = metrics.join(
        dts["category"].value_counts().rename({"count": "n_dts"}),
        on="category",
        how="left",
    ).join(
        gts["category"].value_counts().rename({"count": "n_gts"}),
        on="category",
        how="left",
    )

    # Join median number of interior points
    median_interior_pts = (
        gts.group_by(by="category")
        .agg(pl.col("num_interior_pts").median().cast(pl.UInt32).alias("med_pts"))
        .rename({"by": "category"})
    )

    metrics = metrics.join(
        median_interior_pts,
        on="category",
        how="left",
    )

    metrics = (
        metrics.fill_null(strategy="zero")
        .filter((pl.col("n_gts") > 0) | (pl.col("category") == "AVERAGE_METRICS"))
        .sort("n_gts", descending=True)
    )

    return metrics
