"""Classes which store network outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from torch import Tensor

from torchbox3d.structures.meta import TensorStruct


@dataclass
class NetworkOutputs(TensorStruct):
    """Class which manipulates and tracks data in the `CenterPoint` method.

    Args:
        backbone: (B,C,H,W) Network output from the backbone.
        neck: (B,C,H,W) Network output from the neck.
        head: (T,) Task outputs from the head of the network.
    """

    backbone: Tensor
    neck: Tensor
    head: List[TaskOutputs]


@dataclass
class TaskOutputs(TensorStruct):
    """Task head outputs from the network.

    Args:
        logits: (B,C,H,W) Raw class values before being mapped to [0,1].
        regressands: (B,R,H,W) Tensor of predicted continuous values.
    """

    logits: Tensor
    regressands: Tensor
