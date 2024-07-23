"""Launch a training job."""

import logging
import os
import signal
import sys
import traceback
import warnings
from pathlib import Path
from typing import Final, cast

import hydra
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torchbox3d.utils.hydra import configure_job

warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="pytorch_lightning")
warnings.filterwarnings("ignore", module="submitit")

logger = logging.getLogger(__name__)

HYDRA_PATH: Final = Path(__file__).resolve().parent.parent / "conf"


@hydra.main(
    config_path=str(HYDRA_PATH),
    config_name="config",
    version_base=None,
)
def train(cfg: DictConfig) -> None:
    """Training entrypoint.

    Args:
        cfg: Training configuration.
    """
    # Clear signal to allow lightning to intervene.
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)

    # Create root directory.
    default_root_dir = Path(cfg.trainer.default_root_dir)
    default_root_dir.mkdir(parents=True, exist_ok=True)

    plugins = None

    hydra_cfg = HydraConfig()
    mode = hydra_cfg.cfg["hydra"]["mode"]
    if mode == RunMode.MULTIRUN:
        plugins = SLURMEnvironment(requeue_signal=signal.SIGUSR2)

    # Set start method to forkserver to avoid fork related issues.
    torch.multiprocessing.set_forkserver_preload(["torch"])
    torch.multiprocessing.set_start_method("forkserver")

    # Utilize tensor cores to speed up training.
    torch.set_float32_matmul_precision("medium")

    # Set job options if debugging.
    configure_job(cfg)

    # Manually cast since instantiate isn't typed.
    trainer = cast(
        Trainer,
        hydra.utils.instantiate(
            cfg.trainer, plugins=plugins, enable_checkpointing=not cfg.model.debug
        ),
    )
    datamodule = cast(LightningDataModule, instantiate(cfg.dataset))
    model = cast(
        LightningModule, instantiate(cfg.model, num_devices=trainer.num_devices)
    )

    # if trainer.precision == "bf16-mixed":
    #     trainer.strategy.ddp_comm_hook = default.bf16_compress_hook
    # elif trainer.precision == "16-mixed":
    #     trainer.strategy.ddp_comm_hook = default.fp16_compress_hook

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.log_hyperparams(cfg.trainer)
        # trainer.logger.watch(model, log_graph=True)

    try:
        # trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
        trainer.fit(model, datamodule)
        # trainer.fit(model, train_dataloaders=datamodule.train_dataloader())
        # trainer.validate(model, dataloaders=datamodule.val_dataloader())

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

    # Prevent locking during multiruns.
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass
    os.environ.pop("LOCAL_RANK", None)

    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.finish()


if __name__ == "__main__":
    train()
