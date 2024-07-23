"""PyTorch implementation of an Argoverse 2 (AV2), 3D detection dataloader."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Union

import kornia
import numpy as np
import polars as pl
import torch
import torchvision.transforms.functional as F
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from polars import col
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import torchbox3d._rust as rust
from torchbox3d.datasets.argoverse.constants import (
    LASER_MAPPING,
    LOG_IDS,
    ROW_MAPPING_32,
    ROW_MAPPING_64,
)
from torchbox3d.math.range_view import build_range_view
from torchbox3d.structures.cuboids import Cuboids
from torchbox3d.structures.data import Data, RegularGridData
from torchbox3d.structures.grid import RegularGrid
from torchbox3d.structures.sparse_tensor import SparseTensor
from torchbox3d.structures.targets import GridTargets
from torchbox3d.utils.collater import collate_fn

logger = logging.getLogger(__name__)

ANNOTATION_COLUMNS: Final = (
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
)


@dataclass
class DataLoader(Dataset[Data]):
    """General rust-based dataloader for AV2 formatted datasets.

    Args:
        dataloader: Rust-based dataloader (with python bindings).
        tasks: Task index to task category mapping.
        preprocessing_config: Data input preprocessing configuration.
        target_encoding_config: Target encoding configuration for training.
    """

    _dataloader: DictConfig = MISSING
    tasks: DictConfig = MISSING
    preprocessing_config: Dict[str, Any] = MISSING
    target_encoding_config: Dict[str, Any] = MISSING
    enable_horizontal_flip: bool = MISSING

    def __post_init__(self) -> None:
        """Initialize the data-loader."""
        self.dataloader = instantiate(self._dataloader)

    def __len__(self) -> int:
        """Return the length of the dataset records."""
        return self.dataloader.__len__()

    def _build_preprocessing_config(self):
        """Build preprocessing configutation for the Rust dataloader.

        Rust Enums must be manually built due to limitations in the Rust bindings.
        """
        preprocessing_config = self.preprocessing_config.copy()
        preprocessing_config["projection_type"] = rust.ProjectionType(
            preprocessing_config.pop("projection_type")
        )
        if preprocessing_config["projection_type"] == rust.ProjectionType.BirdsEyeView:
            return rust.BirdsEyeViewPreprocessingConfig(**preprocessing_config)
        elif preprocessing_config["projection_type"] == rust.ProjectionType.RangeView:
            preprocessing_config["buffering_mode"] = rust.BufferingMode(
                preprocessing_config.pop("buffering_mode")
            )
            preprocessing_config["projection_mode"] = rust.ProjectionMode(
                preprocessing_config.pop("projection_mode")
            )
            preprocessing_config["dataset_name"] = rust.DatasetName(
                preprocessing_config.pop("dataset_name")
            )
            return rust.RangeViewPreprocessingConfig(**preprocessing_config)
        else:
            raise NotImplementedError()

    def _build_target_encoding_config(self):
        """Build preprocessing configutation for the Rust dataloader.

        Rust Enums must be manually built due to limitations in the Rust bindings.
        """
        projection_type = self.preprocessing_config["projection_type"]
        target_encoding_config = self.target_encoding_config.copy()
        target_encoding_config["target_encoding_type"] = rust.TargetEncodingType(
            target_encoding_config.pop("target_encoding_type")
        )
        if projection_type == "BIRDS_EYE_VIEW":
            return rust.BirdsEyeViewTargetEncodingConfig(**target_encoding_config)
        elif projection_type == "RANGE_VIEW":
            return rust.RangeViewTargetEncodingConfig(**target_encoding_config)
        else:
            raise NotImplementedError()

    def __getitem__(self, index: int) -> Data:
        """Load an item of the dataset and return it.

        Args:
            index: The dataset item index.

        Returns:
            An item of the dataset.
        """
        sweep = self.dataloader.get(index)
        preprocessing_config = self._build_preprocessing_config()
        target_encoding_config = self._build_target_encoding_config()
        cells: Union[SparseTensor, torch.Tensor]
        if preprocessing_config.projection_type == rust.ProjectionType.RangeView:
            (
                features_npy,
                likelihoods,
                encodings,
                offsets,
                panoptics,
                weights,
            ) = sweep.build_range_view_with_targets(
                dict(self.tasks),
                preprocessing_config,
                target_encoding_config,
            )

            laser_numbers = sweep.lidar["laser_number"].to_numpy().copy()
            (log_id, timestamp_ns) = sweep.sweep_uuid
            num_lasers = 64

            if log_id in LOG_IDS:
                laser_numbers[laser_numbers >= 32] = LASER_MAPPING[
                    laser_numbers[laser_numbers >= 32] - 32
                ]
                laser_numbers[laser_numbers < 32] = LASER_MAPPING[
                    laser_numbers[laser_numbers < 32]
                ]
            if num_lasers == 32:
                laser_mapping = ROW_MAPPING_32.copy()
            else:
                laser_mapping = ROW_MAPPING_64.copy()

            lidar_offset = np.array([1.356, 0.0, 1.726])
            features_npy = build_range_view(
                sweep.lidar,
                laser_mapping=laser_mapping,
                lidar_offset=lidar_offset,
                num_lasers=num_lasers,
            )

            num_features = preprocessing_config.num_features
            features = torch.as_tensor(features_npy, dtype=torch.float32)[:num_features]
            scores = torch.as_tensor(likelihoods)
            encoding = torch.as_tensor(encodings)
            offsets = torch.as_tensor(offsets)
            panoptic = torch.as_tensor(panoptics)
            weights = torch.as_tensor(weights)
            if self.enable_horizontal_flip:
                if torch.rand(1) < 0.5:
                    augmentation = F.hflip
                    features = augmentation(features)
                    scores = augmentation(scores)
                    encoding = augmentation(encoding)
                    offsets = augmentation(offsets)
                    panoptic = augmentation(panoptic)
                    weights = augmentation(weights)

                # if torch.rand(1) < 0.5:
                #     import torchvision

                #     _, height, width = F.get_dimensions(features)
                #     t = torchvision.transforms.ElasticTransform()
                #     params = t.get_params(t.alpha, t.sigma, [height, width])
                #     augmentation = F.elastic_transform
                #     features = augmentation(features, params)
                #     scores = augmentation(scores, params)
                #     encoding = augmentation(encoding, params)
                #     offsets = augmentation(offsets, params)
                #     panoptic = augmentation(panoptic, params)
                #     weights = augmentation(weights, params)

            cells = torch.as_tensor(features_npy, dtype=torch.float32)
            grid_targets = GridTargets(
                scores=scores,
                encoding=encoding,
                offsets=offsets,
                panoptic=panoptic,
                weights=weights,
            )

            height, width = features.shape[1:]
            indices = kornia.utils.create_meshgrid3d(
                height,
                1,
                width,
                normalized_coordinates=False,
                dtype=torch.long,
            ).flatten(0, 3)
            values = features.permute(0, 2, 1).flatten(1, -1).t()
            counts = torch.ones_like(indices[:, 0:1])
            cells = SparseTensor(values=values, indices=indices, counts=counts)

            annotations = rust.preprocess_annotations(
                sweep.annotations, dict(self.tasks)
            )
            cuboids = self._build_cuboids(annotations)

            grid = RegularGrid(
                (-150.0, -150.0, -10.0), (150.0, 150.0, 10.0), (0.1, 0.1, 0.2)
            )

            datum = RegularGridData(
                coordinates_m=values[:, 3:7],
                counts=torch.ones_like(values[:, :1]),
                cuboids=cuboids,
                uuids=sweep.sweep_uuid,
                values=values,
                grid=grid,
                cells=cells,
                targets=grid_targets,
            )
            datum.range_image = features
        elif preprocessing_config.projection_type == rust.ProjectionType.BirdsEyeView:
            (
                indices,
                values,
                counts,
            ), targets = sweep.build_bev_image_with_targets(
                dict(self.tasks),
                preprocessing_config,
                target_encoding_config,
            )

            grid_targets = GridTargets(
                scores=torch.as_tensor(targets.likelihoods),
                encoding=torch.as_tensor(targets.encodings),
                offsets=torch.as_tensor(targets.offsets),
                panoptic=torch.as_tensor(targets.panoptics),
                weights=torch.as_tensor(targets.weights),
            )

            indices = torch.as_tensor(indices.astype(int), dtype=torch.long)
            values = torch.as_tensor(values, dtype=torch.float32)

            counts = torch.zeros_like(indices)
            cells = SparseTensor(values=values, indices=indices, counts=counts)

            annotations = rust.preprocess_annotations(
                sweep.annotations, dict(self.tasks)
            )
            cuboids = self._build_cuboids(annotations)

            grid = RegularGrid(
                tuple(preprocessing_config.min_world_coordinates_m),
                tuple(preprocessing_config.max_world_coordinates_m),
                delta_m_per_cell=tuple(preprocessing_config.delta_m_per_cell),
            )

            datum = RegularGridData(
                coordinates_m=cells.values[:, :3],
                counts=cells.counts,
                cuboids=cuboids,
                uuids=sweep.sweep_uuid,
                values=cells.values,
                grid=grid,
                cells=cells,
                targets=grid_targets,
            )
        else:
            raise NotImplementedError()
        return datum

    def _build_cuboids(self, annotations: pl.DataFrame) -> Cuboids:
        """Build the ground truth cuboids.

        Args:
            annotations: Ground truth annotations.

        Returns:
            Ground truth cuboids.
        """
        cuboid_params = torch.as_tensor(
            annotations.lazy()
            .select(col(ANNOTATION_COLUMNS))
            .collect()
            .to_numpy()
            .astype(np.float32)
        )

        scores = torch.ones(len(cuboid_params), dtype=torch.float)
        task_offsets = torch.as_tensor(
            annotations["offset"].to_numpy().astype(int),
            dtype=torch.long,
        )
        task_ids = torch.as_tensor(
            annotations["task_id"].to_numpy().astype(int),
            dtype=torch.long,
        )
        cuboids = Cuboids(
            params=cuboid_params,
            scores=scores,
            task_offsets=task_offsets,
            task_ids=task_ids,
        )
        return cuboids


@dataclass
class ArgoverseDataModule(LightningDataModule):
    """Construct an Argoverse datamodule."""

    batch_size: int = MISSING
    num_workers: int = MISSING

    _train_dataset: DictConfig = MISSING
    _val_dataset: DictConfig = MISSING
    _test_dataset: DictConfig = MISSING

    tasks_cfg: Dict[int, List[str]] = MISSING
    src_dir: str = MISSING

    def __post_init__(self) -> None:
        """Initialize the meta-datamodule."""
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = instantiate(self._train_dataset)
        self.val_dataset = instantiate(self._val_dataset)
        self.test_dataset = instantiate(self._test_dataset)

    def train_dataloader(self) -> torch.utils.data.DataLoader[Data]:
        """Return the _train_ dataloader.

        Returns:
            The PyTorch _train_ dataloader.
        """
        dataloader: DataLoader[Data] = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader[Data]:
        """Return the _validation_ dataloader.

        Returns:
            The PyTorch _validation_ dataloader.
        """
        dataloader: DataLoader[Data] = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def predict_dataloader(self) -> torch.utils.data.DataLoader[Data]:
        """Return the _predict_ dataloader.

        Returns:
            The PyTorch _predict_ dataloader.
        """
        dataloader: DataLoader[Data] = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def test_dataloader(self) -> DataLoader[Data]:
        """Return the _validation_ dataloader.

        Returns:
            The PyTorch _validation_ dataloader.
        """
        dataloader: DataLoader[Data] = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader
