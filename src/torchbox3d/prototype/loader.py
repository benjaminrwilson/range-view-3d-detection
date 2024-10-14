"""Prototype loader class."""

import logging
import math
import random
from collections import defaultdict
from dataclasses import MISSING, dataclass, field
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Any, DefaultDict, Dict, Final, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from mmcv.ops.box_iou_rotated import box_iou_rotated
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils import data

import torchbox3d
from torchbox3d.math.numpy.conversions import cart_to_sph, sph_to_cart
from torchbox3d.math.ops.index import unravel_index
from torchbox3d.utils.hydra import flatten
from torchbox3d.utils.polars import polars_to_torch

pl.Config.set_tbl_rows(30)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
)

TCH_COLUMN_NAMES: Final = (
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

ROW_MAPPING_64 = np.array(
    [
        56,
        22,
        42,
        28,
        61,
        30,
        49,
        36,
        40,
        32,
        38,
        45,
        34,
        26,
        53,
        59,
        8,
        1,
        16,
        20,
        12,
        5,
        11,
        15,
        17,
        9,
        24,
        6,
        13,
        3,
        19,
        0,
        7,
        41,
        21,
        35,
        2,
        33,
        14,
        27,
        23,
        31,
        25,
        18,
        29,
        37,
        10,
        4,
        55,
        62,
        47,
        43,
        51,
        58,
        52,
        48,
        46,
        54,
        39,
        57,
        50,
        60,
        44,
        63,
    ]
)

AV2_DATAFRAME_COLUMNS: Final = ("timedelta_ns", "intensity", "range", "x", "y", "z")
WAYMO_DATAFRAME_COLUMNS: Final = ("elongation", "intensity", "range", "x", "y", "z")

RANGE_COLUMN: Final = ("range",)
CART_COLUMNS: Final = ("x", "y", "z")


@dataclass
class DataModule(LightningDataModule):
    """Construct an Argoverse datamodule."""

    batch_size: int = MISSING
    num_workers: int = MISSING
    dataset_name: str = MISSING
    root_dir: str = MISSING
    debug: bool = MISSING

    _train_dataset: DictConfig = MISSING
    _val_dataset: DictConfig = MISSING
    _test_dataset: DictConfig = MISSING

    def __post_init__(self) -> None:
        """Initialize the meta-datamodule."""
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters(
            flatten(self._train_dataset, parent_key="train"),
        )
        self.save_hyperparameters(
            flatten(self._val_dataset, parent_key="val"),
        )
        self.save_hyperparameters(
            flatten(self._test_dataset, parent_key="test"),
        )

        self.train_dataset = instantiate(self._train_dataset)
        self.val_dataset = instantiate(self._val_dataset)
        self.test_dataset = instantiate(self._test_dataset)

    def train_dataloader(self) -> data.DataLoader[Dict[str, Any]]:
        """Return the _train_ dataloader.

        Returns:
            The PyTorch _train_ dataloader.
        """
        dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            shuffle=True,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def val_dataloader(self) -> data.DataLoader[Dict[str, Any]]:
        """Return the _validation_ dataloader.

        Returns:
            The PyTorch _validation_ dataloader.
        """
        dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def predict_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, Any]]:
        """Return the _predict_ dataloader.

        Returns:
            The PyTorch _predict_ dataloader.
        """
        dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader

    def test_dataloader(self) -> data.DataLoader[Dict[str, Any]]:
        """Return the _validation_ dataloader.

        Returns:
            The PyTorch _validation_ dataloader.
        """
        dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
        return dataloader


def _collate_fn(batches: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate single examples into batched input."""
    collated: DefaultDict[List[Tensor]] = defaultdict(list)
    for batch in batches:
        for k, v in batch.items():
            collated[k].append(v)

    for k, v in collated.items():
        elem = v[0]
        if isinstance(elem, pl.DataFrame):
            collated[k] = pl.concat(
                [e.with_columns(batch_index=i) for i, e in enumerate(v)]
            )
        else:
            collated[k] = torch.stack(v)
    return dict(collated)


@dataclass
class DataLoader:
    """Data and annotation preprocessor."""

    root_dir: str
    dataset_name: str
    split_name: str

    range_view_config: DictConfig = MISSING
    augmentations_config: Optional[DictConfig] = None
    targets_config: Optional[DictConfig] = None
    db_config: Optional[DictConfig] = None
    subsampling_rate: int = 1
    num_frames: int = 1
    x_stride: int = 1
    padding_mode: str = "constant"
    use_median_filter: bool = False
    normalize_input: bool = False
    enable_database: bool = False
    use_repeat_factor_sampling: bool = False

    metadata: pl.DataFrame = field(init=False)

    db: Optional[Dict[str, pl.DataFrame]] = field(init=False)

    def __post_init__(self) -> None:
        """Post initialization."""
        self._build_index()
        self._build_annotations_metadata()
        if self.enable_database:
            self._load_db()
        else:
            self.db = None
        self.metadata = self.metadata[:: self.subsampling_rate]
        # self.metadata = self.metadata[:1]

    def _load_db(self) -> None:
        logger.info("Loading database ...")
        file_path = Path(self.root_dir).parent / "db" / "db.feather"
        db = pl.scan_ipc(file_path).filter(pl.col("num_interior_pts") > 0).collect()
        self.db = db.partition_by("category", as_dict=True)

    def _build_index(self) -> None:
        log_ids = []
        timestamp_nss = []

        for log_path in sorted(self.split_dir.glob("*")):
            log_id = log_path.stem
            sweep_paths = (log_path / "sensors" / "lidar").glob("*.feather")
            timestamps = [int(sweep_path.stem) for sweep_path in sorted(sweep_paths)]

            log_ids.extend([log_id] * len(timestamps))
            timestamp_nss.extend(timestamps)
        self.metadata = pl.DataFrame(
            {
                "log_id": np.array(log_ids),
                "timestamp_ns": np.array(timestamp_nss, dtype=np.uint64),
            }
        )

    def _build_annotations_metadata(self) -> None:
        annotations_list: List[pl.LazyFrame] = []
        for log_path in sorted(self.split_dir.glob("*")):
            annotations_path = log_path / "annotations.feather"
            annotations = pl.read_ipc(annotations_path)
            annotations = annotations.with_columns(
                pl.Series("log_id", [log_path.stem] * len(annotations))
            )

            if len(annotations) > 0:
                annotations_list.append(annotations.lazy())

        if len(annotations_list) > 0:
            annotations_lazy = pl.concat(annotations_list).cast(
                {"timestamp_ns": pl.UInt64}
            )

        if self.split_name in ("train",):
            categories = tuple(
                chain.from_iterable(self.targets_config["tasks"].values())
            )

            # Filter scenes without objects of interest.
            # Filter scenes with no objects.
            annotations_lazy = annotations_lazy.filter(
                (pl.col("category").is_in(categories)) & (pl.col("num_interior_pts"))
                > 0
            ).join(
                self.metadata.lazy().select(("log_id", "timestamp_ns")),
                on=("log_id", "timestamp_ns"),
            )

            # Waymo contains "no-label-zone" (NLZ) regions which are masked out of the range images.
            # This causes issues in normalization if normalizing by the number of foreground examples because some range images contain very few points which can cause training to diverge.
            # We filter range images in the training set that have less than 50000 points.
            # It should be noted that the majority of range images have many more points. The max number of points is 64 * 2650 (the total number of pixels).
            if self.dataset_name == "waymo":
                metadata_dir = (
                    Path(torchbox3d.__file__).parent.parent.parent / "metadata"
                )
                metadata = pl.read_ipc(metadata_dir / "waymo.feather")
                metadata_lazy = metadata.filter(pl.col("num_pts") >= 50000).lazy()
                annotations_lazy = annotations_lazy.join(
                    metadata_lazy, on=("log_id", "timestamp_ns")
                )

            uuids = annotations_lazy.select(("log_id", "timestamp_ns")).unique(
                maintain_order=True
            )
            self.metadata = (
                self.metadata.lazy()
                .join(uuids, on=("log_id", "timestamp_ns"))
                .collect()
            )

            if self.use_repeat_factor_sampling:
                annotations = annotations_lazy.collect().sort(
                    ("log_id", "timestamp_ns")
                )
                rfs = (
                    annotations.select(("log_id", "timestamp_ns", "category"))
                    .unique()["category"]
                    .value_counts()
                )
                rfs = rfs.with_columns(f_c=pl.col("count") / pl.col("count").sum())

                t = 0.01
                rfs = rfs.with_columns(
                    r_c=pl.max_horizontal(1, (t / pl.col("f_c")).sqrt())
                )

                r_i = (
                    annotations.join(rfs, "category")
                    .group_by(("log_id", "timestamp_ns"), maintain_order=True)
                    .agg(pl.col("r_c").max())
                )

                r_i = r_i.with_columns(int_part=pl.col("r_c").apply(np.trunc))
                r_i = r_i.with_columns(frac_part=pl.col("r_c") - pl.col("int_part"))

                torch.manual_seed(0)
                thresholds = torch.rand(r_i.shape[0])
                r_i = r_i.with_columns(random=thresholds.numpy())
                r_i = r_i.with_columns(
                    rep_factor=pl.col("int_part")
                    + (pl.col("random") < pl.col("frac_part")).cast(pl.Int32)
                ).sort(("log_id", "timestamp_ns"))

                index = torch.arange(0, r_i.shape[0])
                indices = index.repeat_interleave(
                    torch.as_tensor(r_i["rep_factor"]).long(), 0
                ).numpy()

                samples = self.metadata[indices]
                old_distribution = (
                    self.metadata.join(annotations, on=("log_id", "timestamp_ns"))[
                        "category"
                    ]
                    .value_counts()
                    .rename({"count": "old_count"})
                )
                new_distribution = (
                    samples.join(annotations, on=("log_id", "timestamp_ns"))["category"]
                    .value_counts()
                    .rename({"count": "new_count"})
                )

                distributions = (
                    old_distribution.join(new_distribution, on="category")
                    .sort("category")
                    .with_columns(diff=pl.col("new_count") - pl.col("old_count"))
                )
                logger.info(
                    "Training distribution:%s",
                    distributions,
                )
                self.metadata = samples.select(("log_id", "timestamp_ns"))
                self.metadata = self.metadata[indices]

                # print(self.metadata.shape)

                # r_i.with_row_count()["row_nr"]
                # r_i["rep_factor"]

                # count = annotations["category"].value_counts()
                # count = count.with_columns(inverse=(1 / count["count"]) ** 0.25)
                # count = count.with_columns(
                #     inverse=count["inverse"] / count["inverse"].sum()
                # )

                # samples_list = []
                # for category, _, ratio in count.rows():
                #     num_samples = self.metadata.shape[0] * ratio
                #     samples = annotations.filter(pl.col("category") == category).sample(
                #         num_samples, with_replacement=True
                #     )
                #     samples_list.append(samples)
                # samples = pl.concat(samples_list)

                # logger.info(
                #     "Training distribution:%s",
                #     samples["category"].value_counts().sort("category"),
                # )
                # self.metadata = samples.select(("log_id", "timestamp_ns"))
            else:
                count = (
                    annotations_lazy.collect()["category"]
                    .value_counts()
                    .sort("category")
                )
                count = count.with_columns(
                    percentage=(count["count"] / count["count"].sum()).round(3)
                )
                logger.info(
                    "Training distribution: %s",
                    count,
                )

    def __len__(self) -> int:
        """Number of sweeps in the split."""
        return len(self.metadata)

    @property
    def split_dir(self) -> Path:
        """Dataset split directory."""
        return Path(self.root_dir) / self.split_name

    def annotations_path(self, log_id: str) -> Path:
        """Annotations path for a given log id."""
        return Path(self.root_dir) / self.split_name / log_id / "annotations.feather"

    def lidar_path(self, log_id: str, timestamp_ns: int) -> Path:
        """Annotations path for a given log id."""
        return (
            Path(self.root_dir)
            / self.split_name
            / log_id
            / "sensors"
            / "range_view"
            / f"{timestamp_ns}.feather"
        )

    @property
    def lidar_column_names(self) -> Tuple[str, ...]:
        """Lidar feature column names."""
        if self.dataset_name == "av2" or self.dataset_name == "nuscenes":
            return AV2_DATAFRAME_COLUMNS
        elif self.dataset_name == "waymo":
            return WAYMO_DATAFRAME_COLUMNS
        else:
            raise NotImplementedError(f"{self.dataset_name} is not implemented.")

    def _point_dropout(self, sweep_pl: pl.LazyFrame, p: float) -> pl.LazyFrame:
        features = sweep_pl.collect()
        mask = np.random.rand(features.shape[0], 1) <= p

        features_npy = features.to_numpy() * mask
        sweep_pl = pl.from_numpy(data=features_npy, schema=sweep_pl.schema).lazy()
        return sweep_pl

    def apply_augmentations(
        self, sweep_pl: pl.LazyFrame, annotations: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        if self.augmentations_config is not None:
            for k, v in self.augmentations_config.items():
                if k == "point_dropout":
                    sweep_pl = self._point_dropout(sweep_pl, **v)
                elif k == "flip_azimuth":
                    sweep_pl, annotations = flip_azimuth(
                        sweep_pl,
                        annotations,
                        self.range_view_config,
                        **v,
                    )
                elif k == "random_rotation":
                    sweep_pl, annotations = random_rotation(
                        sweep_pl,
                        annotations,
                        self.range_view_config,
                        **v,
                    )
                elif k == "random_global_scale":
                    sweep_pl, annotations = random_global_scale(
                        sweep_pl,
                        annotations,
                        **v,
                    )
                elif k == "random_global_translation":
                    sweep_pl, annotations = random_global_translation(
                        sweep_pl,
                        annotations,
                        **v,
                    )
        else:
            raise RuntimeError("Augmentations configuration must exist.")
        return sweep_pl, annotations

    @cached_property
    def categories(self) -> Tuple[str, ...]:
        categories = tuple(chain.from_iterable(self.targets_config["tasks"].values()))
        return categories

    @cached_property
    def tasks_frame(self) -> pl.LazyFrame:
        task_data = defaultdict(list)
        for k, task_categories in self.targets_config["tasks"].items():
            for offset, category in enumerate(sorted(task_categories)):
                task_data["task_id"].append(k)
                task_data["offset"].append(offset)
                task_data["category"].append(category)

        task_frame = pl.DataFrame(task_data)
        return task_frame.lazy()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get an item from the dataset."""
        uuid = (
            self.metadata.lazy()
            .with_row_count()
            .filter(pl.col("row_nr") == index)
            .select(("log_id", "timestamp_ns"))
        ).collect()

        # Sanity check.
        assert uuid.shape[0] == 1
        log_id, timestamp_ns = uuid.row(0)

        # Build annotation filter and load annotations lazily.
        mask = (
            (pl.col("timestamp_ns") == timestamp_ns)
            & (pl.col("num_interior_pts") > 0)
            & (pl.col("category").is_in(self.categories))
        )
        annotations = pl.scan_ipc(self.annotations_path(log_id)).filter(mask)
        datum: Dict[str, Union[pl.DataFrame, Tensor]] = {"uuids": uuid}

        features_column_names = self.range_view_config["feature_column_names"]
        sweep_pl = pl.scan_ipc(self.lidar_path(log_id, timestamp_ns))

        # Filter ROI for supported datasets.
        if self.range_view_config.get("filter_roi", False):
            sweep_eager = sweep_pl.collect()
            sweep_pl = (sweep_eager * sweep_eager["is_within_roi"]).lazy()

        # Apply augmentations.
        if self.split_name == "train":
            if self.augmentations_config is not None:
                sweep_pl, annotations = self.apply_augmentations(
                    sweep_pl=sweep_pl, annotations=annotations
                )

        if "view" in features_column_names:
            rev_mapping = {v: i for i, v in enumerate(ROW_MAPPING_64)}

            range_expr = pl.col("range").gt(0).cast(pl.Float32)
            sweep_pl = sweep_pl.with_columns(
                laser_number=pl.col("laser_number").replace(rev_mapping) * range_expr
            )

            sweep_pl = sweep_pl.with_columns(
                view=(
                    (
                        2 * (pl.col("laser_number") <= 32).cast(pl.Float32)
                        + (pl.col("laser_number") > 32).cast(pl.Float32)
                    )
                    * range_expr
                )
            )

        features_pl = sweep_pl.select(features_column_names)

        if self.dataset_name == "waymo":
            features_pl = features_pl.with_columns(intensity=pl.col("intensity").tanh())
        cart_pl = sweep_pl.select(CART_COLUMNS)
        range_pl = sweep_pl.select("range")

        if "timedelta_ns" in features_column_names:
            features_pl = features_pl.with_columns(
                timedelta_ns=pl.col("timedelta_ns") * 1e-9
            )

        h, w = self.range_view_config["height"], self.range_view_config["width"]
        num_features = len(features_column_names)
        features = _npy_to_tch(
            features_pl.collect().to_numpy(writable=True),
            (num_features, h, w),
            torch.float32,
        )
        cart = _npy_to_tch(
            cart_pl.collect().to_numpy(writable=True), (3, h, w), torch.float32
        )
        mask = (
            _npy_to_tch(
                range_pl.collect().to_numpy(writable=True), (1, h, w), torch.float32
            )
            > 0.0
        )

        # if self.dataset_name == "waymo":
        #     features = _normalize_waymo(features)

        # import torch.nn.functional as F

        # features = F.interpolate(
        #     features.unsqueeze(0), scale_factor=(2, 1), mode="nearest-exact"
        # ).squeeze(0)
        # cart = F.interpolate(
        #     cart.unsqueeze(0), scale_factor=(2, 1), mode="nearest-exact"
        # ).squeeze(0)
        # mask = cart.norm(dim=0, keepdim=True) > 0
        # mask = (
        #     F.interpolate(
        #         mask.float().unsqueeze(0), scale_factor=(2, 1), mode="nearest-exact"
        #     )
        #     .bool()
        #     .squeeze(0)
        # )

        if self.enable_database:
            if self.db_config is not None:
                root_dir = Path(self.root_dir).parent / "db" / "train"
                annotations, features, cart, mask = sample_database(
                    root_dir=root_dir,
                    database=self.db,
                    database_config=self.db_config,
                    annotations=annotations,
                    range_view=features,
                    cart=cart,
                    range_mask=mask,
                    lidar_column_names=self.range_view_config["feature_column_names"],
                )
            else:
                raise RuntimeError("Database config must be defined.")

        features, mask, cart = subsample_range_view(
            features,
            mask,
            cart,
            self.dataset_name,
            self.x_stride,
            self.padding_mode,
        )
        datum["features"] = features
        datum["mask"] = mask
        datum["cart"] = cart
        if self.targets_config is not None:
            datum["annotations"] = (
                annotations.join(self.tasks_frame, on="category")
                .sort(["task_id", "offset"])
                .collect()
            )
        return datum


def sample_database(
    root_dir: Path,
    database: Dict[str, pl.DataFrame],
    database_config: DictConfig,
    annotations: pl.LazyFrame,
    range_view: Tensor,
    cart: Tensor,
    range_mask: Tensor,
    lidar_column_names: Tuple[str, ...],
) -> Tuple[pl.DataFrame, Tensor, Tensor, Tensor]:
    samples_list: List[pl.DataFrame] = []
    for k, num_samples in database_config.items():
        db_annotations = database[k]

        num_samples = min(db_annotations.shape[0], num_samples)
        if num_samples > 0:
            samples = db_annotations.sample(num_samples)
            samples_list.append(samples)

    db_samples = pl.concat(samples_list)
    ious = _intersection_test(annotations.collect(), db_samples)
    mask = (ious > 0).sum(dim=0) == 0

    db_samples = db_samples.filter(pl.lit(mask.numpy()))

    ious = _intersection_test(db_samples, db_samples)
    mask = (ious > 0).sum(dim=0) == 1
    db_samples = db_samples.filter(pl.lit(mask.numpy()))

    _, height, width = range_view.shape
    features_list: List[pl.DataFrame] = []
    for row in db_samples.select(pl.col(["category", "row_nr"])).rows():
        category, row_nr = row[0], row[1]
        file_path = root_dir / category / f"{row_nr}.feather"
        samples_frame = pl.read_ipc(file_path).with_columns(row_nr=row_nr)
        features_list.append(samples_frame)

    samples_frame = pl.concat(features_list).sort("range")
    samples_frame = samples_frame.unique("index", keep="first")

    valid_nr = samples_frame.select(pl.col("row_nr")).unique().to_series()
    db_samples = db_samples.filter(pl.col("row_nr").is_in(valid_nr))

    features = torch.as_tensor(samples_frame.select(lidar_column_names).to_numpy())
    samples_cart = torch.as_tensor(samples_frame.select(["x", "y", "z"]).to_numpy())
    samples_mask = torch.as_tensor(np.linalg.norm(samples_cart, axis=-1) > 0.0)

    indices = torch.as_tensor(samples_frame.select("index").to_numpy()).squeeze(-1)

    _, height, width = range_view.shape
    indices = unravel_index(
        indices,
        shape=[
            height,
            width,
        ],
    )
    rows, cols = indices.t()
    range_view[:, rows, cols] = features.t()
    cart[:, rows, cols] = samples_cart.t()
    range_mask[:, rows, cols] = samples_mask.t()
    range_view *= range_mask

    annotations = pl.concat([annotations, db_samples.drop(["log_id", "row_nr"]).lazy()])
    return annotations, range_view, cart, range_mask


def _intersection_test(annotations: pl.DataFrame, db_samples: pl.DataFrame) -> Tensor:
    annotations_tch = polars_to_torch(
        annotations=annotations,
        columns=TCH_COLUMN_NAMES,
        device="cpu",
    ).type(torch.float32)
    db_samples_tch = polars_to_torch(
        db_samples, columns=TCH_COLUMN_NAMES, device="cpu"
    ).type(torch.float32)

    ious = box_iou_rotated(
        annotations_tch[:, [0, 1, 3, 4, -1]],
        db_samples_tch[:, [0, 1, 3, 4, -1]],
    )
    return ious


def subsample_range_view(
    range_view: Tensor,
    mask: Tensor,
    cart: Tensor,
    dataset_name: str,
    x_stride: int,
    mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    if dataset_name == "waymo":
        if x_stride == 4:
            pad = [19, 19]
        else:
            pad = [3, 3]
    elif dataset_name == "av2":
        if x_stride == 4:
            pad = [28, 28]
        else:
            pad = [4, 4]

    range_view *= mask
    range_view = torch.nn.functional.pad(range_view, pad, mode=mode)[:, :, ::x_stride]
    mask = torch.nn.functional.pad(mask, pad, mode=mode)[:, :, ::x_stride]
    cart = torch.nn.functional.pad(cart, pad, mode=mode)[:, :, ::x_stride]
    return range_view, mask, cart


def _npy_to_tch(npy: np.ndarray, shape: Tuple[int, ...], dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(
        npy.transpose(1, 0).reshape(*shape),
        dtype=dtype,
    )


def random_rotation(
    sweep_lazy: pl.LazyFrame,
    annotations_lazy: pl.LazyFrame,
    range_view_config: DictConfig,
    low: float,
    high: float,
    p: float,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Rotate the range image and associated features about the azimuth."""
    if random.random() > p:
        return sweep_lazy, annotations_lazy
    theta = random.uniform(low, high)

    height, width = range_view_config["height"], range_view_config["width"]
    shift = math.floor(theta / math.tau * width)

    sweep = sweep_lazy.collect()
    img = sweep.to_numpy().transpose(1, 0).reshape(-1, height, width)
    img = np.roll(img, shift=shift, axis=-1)
    sweep = pl.from_numpy(
        img.reshape(-1, height * width).transpose(1, 0), schema=sweep.schema
    )

    rot = np.array(
        [
            math.cos(theta),
            -math.sin(theta),
            0.0,
            math.sin(theta),
            math.cos(theta),
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ).reshape(3, 3)

    cart = sweep.select(("x", "y", "z")).to_numpy().transpose(1, 0)

    x, y, z = rot.T @ cart
    sweep = sweep.with_columns(x=x, y=y, z=z)

    annotations = annotations_lazy.collect()
    if annotations.shape[0] > 0:
        cuboids = annotations.select(CUBOID_COLUMN_NAMES).to_numpy().transpose(1, 0)

        x, y, z = rot.T @ cuboids[:3]
        annotations = annotations.with_columns(tx_m=x, ty_m=y, tz_m=z)

        quat = annotations.select(("qx", "qy", "qz", "qw")).to_numpy()
        mat = Rotation.from_quat(quat).as_matrix()
        mat = mat @ rot.T

        qx, qy, qz, qw = Rotation.from_matrix(mat).as_quat().T
        annotations = annotations.with_columns(qx=qx, qy=qy, qz=qz, qw=qw)
    return sweep.lazy(), annotations.lazy()


def random_global_scale(
    sweep_lazy: pl.LazyFrame,
    annotations_lazy: pl.LazyFrame,
    low: float,
    high: float,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Rotate the range image and associated features about the azimuth."""
    sweep = sweep_lazy.collect()

    scale = random.uniform(low, high)
    sweep = sweep.with_columns(
        x=scale * pl.col("x"), y=scale * pl.col("y"), z=scale * pl.col("z")
    )

    new_range = np.linalg.norm(sweep.select(["x", "y", "z"]).to_numpy(), 2, axis=-1)
    sweep = sweep.with_columns(range=new_range)

    annotations = annotations_lazy.collect()
    if annotations.shape[0] > 0:
        annotations = annotations.with_columns(
            tx_m=scale * pl.col("tx_m"),
            ty_m=scale * pl.col("ty_m"),
            tz_m=scale * pl.col("tz_m"),
            length_m=scale * pl.col("length_m"),
            width_m=scale * pl.col("width_m"),
            height_m=scale * pl.col("height_m"),
        )

    return sweep.lazy(), annotations.lazy()


def random_global_translation(
    sweep_lazy: pl.LazyFrame,
    annotations_lazy: pl.LazyFrame,
    std_x: float,
    std_y: float,
    std_z: float,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Rotate the range image and associated features about the azimuth."""
    sweep = sweep_lazy.collect()

    x = random.normalvariate(0, std_x)
    y = random.normalvariate(0, std_y)
    z = random.normalvariate(0, std_z)

    sweep = sweep.with_columns(x=x + pl.col("x"), y=y + pl.col("y"), z=z + pl.col("z"))

    annotations = annotations_lazy.collect()
    if annotations.shape[0] > 0:
        annotations = annotations.with_columns(
            tx_m=x + pl.col("tx_m"),
            ty_m=y + pl.col("ty_m"),
            tz_m=z + pl.col("tz_m"),
        )

    return sweep.lazy(), annotations.lazy()


def flip_azimuth(
    sweep_lazy: pl.LazyFrame,
    annotations_lazy: pl.LazyFrame,
    range_view_config: DictConfig,
    p: float,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Flip azimuth of the range image and annotations."""
    if random.random() > p:
        return sweep_lazy, annotations_lazy
    sweep = sweep_lazy.collect()

    height, width = range_view_config["height"], range_view_config["width"]
    img = sweep.to_numpy().transpose(1, 0).reshape(-1, height, width)
    img = np.flip(img, axis=2)

    sweep = pl.from_numpy(
        img.reshape(-1, height * width).transpose(1, 0), schema=sweep.schema
    )

    cart = sweep.select(("x", "y", "z")).to_numpy()
    sph = cart_to_sph(cart)
    sph[:, 0] *= -1

    x, y, z = sph_to_cart(sph).T
    assert np.allclose(cart[:, -1], z)
    sweep = sweep.with_columns(x=x, y=y, z=z)

    annotations = annotations_lazy.collect()
    if annotations.shape[0] > 0:
        cuboids = annotations.select(CUBOID_COLUMN_NAMES).to_numpy()
        cart = annotations.select(("tx_m", "ty_m", "tz_m")).to_numpy()
        sph = cart_to_sph(cart)
        sph[:, 0] *= -1

        x, y, z = sph_to_cart(sph).T
        assert np.allclose(cart[:, -1], z)
        annotations = annotations.with_columns(tx_m=x, ty_m=y, tz_m=z)

        rot = -Rotation.from_quat(cuboids[:, 6:10]).as_euler("xyz")[:, -1]

        # Constrain to [-pi, pi).
        # rot = rot - np.floor(rot / (2 * np.pi) + 0.5) * 2 * np.pi

        xyz = np.zeros((len(rot), 3))
        xyz[:, -1] = rot
        quat = Rotation.from_euler("xyz", xyz, degrees=False).as_quat()

        qx, qy, qz, qw = quat.T
        annotations = annotations.with_columns(qx=qx, qy=qy, qz=qz, qw=qw)
    return sweep.lazy(), annotations.lazy()
