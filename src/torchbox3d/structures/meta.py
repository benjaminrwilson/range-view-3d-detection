"""Detection data classes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, ItemsView, Sequence, Type, TypeVar

import torch
from torch import Tensor

T = TypeVar("T", bound="TensorStruct")


@dataclass
class TensorStruct:
    """Meta-structure which provides common functionality for tensors."""

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the attribute names and values."""
        return ItemsView({k: v for k, v in self.__dict__.items()})

    def cpu(self: T) -> T:
        """Move all tensors to the cpu."""
        for k, v in self.items():
            if isinstance(v, Tensor):
                setattr(self, k, v.cpu())
        return self

    @classmethod
    def cat(cls: Type[T], data_list: Sequence[T]) -> T:
        """Concatenate all of the tensors in the object.

        Args:
            data_list: List of T.

        Returns:
            The concatenated T list.
        """
        output: DefaultDict[str, Any] = defaultdict(list)
        for datum in data_list:
            for key, value in datum.items():
                if value is not None:
                    output[key].append(value)
        kwargs = {k: torch.cat(v, dim=0) for k, v in output.items()}
        return cls(**kwargs)

    @classmethod
    def stack(cls: Type[T], data_list: Sequence[T]) -> T:
        """Stack all of the tensors in the object.

        Args:
            data_list: List of T.

        Returns:
            The concatenated T list.
        """
        output: DefaultDict[str, Any] = defaultdict(list)
        for datum in data_list:
            for key, value in datum.items():
                if value is not None:
                    output[key].append(value)
        kwargs = {k: torch.stack(v, dim=0) for k, v in output.items()}
        return cls(**kwargs)
