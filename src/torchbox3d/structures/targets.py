"""Detection data classes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import DefaultDict, List, Mapping, TypeVar

import torch
from torch import Tensor

from torchbox3d.structures.meta import TensorStruct

T = TypeVar("T", bound="Targets")


@dataclass
class Targets(TensorStruct):
    """Models general network targets."""

    scores: Tensor
    encoding: Tensor


@dataclass
class GridTargets(Targets):
    """Class which manipulates and tracks data in the `CenterPoint` method."""

    scores: Tensor
    encoding: Tensor
    offsets: Tensor
    panoptic: Tensor
    weights: Tensor

    @cached_property
    def mask(self) -> Tensor:
        """Validity mask."""
        return self.panoptic > 0


@dataclass
class CenterPointLoss:
    """Centerpoint loss object.

    Args:
        positive_loss:
        negative_loss:
        coordinate_loss:
        dimension_loss:
        rotation_loss
        weight:
    """

    positive_loss: Tensor
    negative_loss: Tensor

    coordinate_loss: Tensor
    dimension_loss: Tensor
    rotation_loss: Tensor

    weight: float

    @property
    def classification_loss(self) -> Tensor:
        """Return the classification loss."""
        return torch.mean(self.positive_loss + self.negative_loss)

    @property
    def regression_loss(self) -> Tensor:
        """Return the regression loss."""
        return (
            self.coordinate_loss.sum(dim=-1)
            + self.dimension_loss.sum(dim=-1)
            + self.rotation_loss.sum(dim=-1)
        ).mean()

    @property
    def loss(self) -> Tensor:
        """Return the loss."""
        return self.classification_loss + self.weight * self.regression_loss

    def as_dict(self) -> Mapping[str, Tensor]:
        """Return the loss as a dictionary."""
        return {
            "losses/total": self.loss.detach(),
            "losses/positive": self.positive_loss.detach().mean(),
            "losses/negative": self.negative_loss.detach().mean(),
            "losses/coordinate": self.coordinate_loss.detach().sum(dim=-1).mean(),
            "losses/dimension": self.dimension_loss.detach().sum(dim=-1).mean(),
            "losses/rotation": self.rotation_loss.detach().sum(dim=-1).mean(),
        }

    @classmethod
    def stack(cls, data_list: List[CenterPointLoss]) -> CenterPointLoss:
        """Stack the the losses and wrap into one object.

        Args:
            data_list: List of CenterPoint losses.

        Returns:
            A single object that contains all of the losses.
        """
        collated_data: DefaultDict[str, List[Tensor]] = defaultdict(list)

        for data in data_list:
            collated_data["positive_loss"].append(data.positive_loss)
            collated_data["negative_loss"].append(data.negative_loss)
            collated_data["coordinate_loss"].append(data.coordinate_loss)
            collated_data["dimension_loss"].append(data.dimension_loss)
            collated_data["rotation_loss"].append(data.rotation_loss)

        output = {
            attr_name: torch.stack(attr, dim=0)
            for attr_name, attr in collated_data.items()
        }
        return cls(**output, weight=data_list[0].weight)

    def reduce_task_loss(self) -> CenterPointLoss:
        """Reduce the task loss (first dimension).

        Returns:
            The reduced loss object.
        """
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.mean(dim=0))
        return self
