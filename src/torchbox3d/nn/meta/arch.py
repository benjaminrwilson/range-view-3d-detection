"""General object detection model."""

import math
from dataclasses import dataclass
from typing import Any, Dict

from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler


@dataclass(unsafe_hash=True)
class MetaDetector(LightningModule):
    """Construct a general detector object.

    Args:
        batch_size: Batch size used to determine training learning rate.
        num_devices: Number of devices being used (used for scaling LR).
        use_linear_lr_scaling: Boolean flag to enable linear lr batch scaling.
        debug: Boolean flag to enable debugging.
        _optimizer: Partial configuration for network optimizer.
        _scheduler: Partial configuration for network scheduler.
        _backbone: Partial configuration for network backbone.
        _head: Partial configuration for network heads.
        _decoder: Partial configuration for network output decoder.
    """

    batch_size: int = MISSING
    num_devices: int = MISSING
    num_workers: int = MISSING
    use_linear_lr_scaling: bool = MISSING
    debug: bool = MISSING

    _optimizer: DictConfig = MISSING
    _scheduler: DictConfig = MISSING
    _backbone: DictConfig = MISSING
    _head: DictConfig = MISSING
    _decoder: DictConfig = MISSING

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.backbone = instantiate(self._backbone)
        self.head = instantiate(self._head)
        self.decoder = instantiate(self._decoder)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizers and learning rate scheduler for training.

        Returns:
            The optimizers and learning rate scheduler.

        Raises:
            RuntimeError: If trainer is `None`.
        """
        optimizer = instantiate(self._optimizer, params=self.parameters())
        lr_dict: Dict[str, Any] = {}
        if not self.debug:
            self._scheduler["total_steps"] = int(
                self.trainer.estimated_stepping_batches
            )
            if self.use_linear_lr_scaling:
                self._scheduler["max_lr"] *= math.sqrt(
                    self.num_devices * self.batch_size
                )
            scheduler = instantiate(self._scheduler, optimizer=optimizer)
            lr_dict = {
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "metrics/lr",
                }
            }
        return {"optimizer": optimizer} | lr_dict
