"""Hydra resolver and utilitiess."""

import logging
import signal
from collections.abc import MutableMapping
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def flatten(
    dictionary: DictConfig, parent_key: str = "", separator: str = "."
) -> Dict[Any, Any]:
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{separator}{key if parent_key else key}"
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def resolve_signal(signal_name: str) -> signal.Signals:
    """Resolve a signal name.

    Args:
        signal_name: A valid signal name.

    Returns:
        Signals objects corresponding to the signal.
    """
    if signal_name == "SIGUSR2":
        return signal.SIGUSR2
    else:
        raise NotImplementedError("This signal is not supported.")


def resolve_symbol(symbol: str) -> float:
    """Resolve a mathematical symbol.

    Args:
        symbol: Mathematical symbol.

    Returns:
        Float representation of the symbol.
    """
    return float(symbol)


def configure_job(cfg: DictConfig) -> None:
    """Configure the job based on the config.

    Args:
        cfg: Training configuration.
    """
    if cfg.model.debug:
        logger.info("Using debug mode ...")
        callbacks = []
        for _, callback in enumerate(cfg.trainer.callbacks):
            if callback["_target_"] == "pytorch_lightning.callbacks.ModelCheckpoint":
                continue
            callbacks.append(callback)
        cfg.trainer.callbacks = ListConfig(callbacks)
        cfg.model.num_workers = 0
        cfg.model.train_log_freq = 20
        cfg.model.val_log_freq = 20


OmegaConf.register_new_resolver("resolve_signal", resolve_signal)
OmegaConf.register_new_resolver("resolve_symbol", resolve_symbol)
