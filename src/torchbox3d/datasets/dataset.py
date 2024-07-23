"""Dataset metaclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """A dataset metaclass.

    Args:
        dataset_dir: Dataset directory.
        dataset_name: Name of the dataset.
        split: Split name for the dataset.
        categories: List of valid classes for targets.
    """

    dataset_dir: str = MISSING
    dataset_name: str = MISSING
    split: str = MISSING
    categories: Optional[Tuple[str, ...]] = None
