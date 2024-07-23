"""Datamodule metaclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision.transforms import Compose


@dataclass
class DataModule(LightningDataModule):
    """Construct a general datamodule object.

    Args:
        train_transforms_cfg: Train transforms config.
        val_transforms_cfg: Val transforms config.
        test_transforms_cfg: Test transforms config.
        dst_dir: The default data directory.
        num_workers: Number of dataloader workers.
        src_dir: The default data directory.
        tasks_cfg: List of tasks.
        batch_size: Dataloader batch size.
        dataset_name: Dataset name.
    """

    train_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]
    val_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]
    test_transforms_cfg: Optional[Dict[str, Callable[..., Any]]]

    dst_dir: Path
    num_workers: int
    src_dir: Path
    tasks_cfg: DictConfig

    batch_size: int
    dataset_name: str

    prepare_data_per_node: bool = False

    _train_transforms: Callable[..., Any] = field(init=False)
    _val_transforms: Callable[..., Any] = field(init=False)
    _test_transforms: Callable[..., Any] = field(init=False)

    def __post_init__(self) -> None:
        """Compose the data transforms (if the configurations exist)."""
        if self.train_transforms_cfg is not None:
            self._train_transforms = Compose(
                [instantiate(x) for x in self.train_transforms_cfg.values()]
            )
        if self.val_transforms_cfg is not None:
            self._val_transforms = Compose(
                [instantiate(x) for x in self.val_transforms_cfg.values()]
            )
        if self.test_transforms_cfg is not None:
            self._test_transforms = Compose(
                [instantiate(x) for x in self.test_transforms_cfg.values()]
            )

        self.save_hyperparameters(
            ignore=[
                "tasks_cfg",
                "train_transforms_cfg",
                "val_transforms_cfg",
                "test_transforms_cfg",
            ]
        )
