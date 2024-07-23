"""Classification losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from torchbox3d.nn.functional import penalty_reduced_focal_loss, varifocal_loss


@torch.jit.script
@dataclass
class VarifocalLoss:
    """Varifocal Loss."""

    alpha: float
    gamma: float
    reduction: str

    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Creates a criterion that computes Varifocal loss.

        Args:
            input: (*), where * means any number of dimensions.
            target: (*), same shape as the input.

        Returns:
            Scalar. If reduction is 'none', then (*), same shape as input.
        """
        loss = varifocal_loss(
            input=input,
            target=target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        # loss = sigmoid_focal_loss(
        #     inputs=input,
        #     targets=target,
        #     alpha=self.alpha,
        #     gamma=self.gamma,
        #     reduction=self.reduction,
        # )
        return loss

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)


@torch.jit.script
@dataclass
class FocalLoss:
    """Focal Loss class."""

    alpha: float
    gamma: int
    reduction: str

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Creates a criterion that computes Focal loss.

        Args:
            input: (*), where * means any number of dimensions.
            target: (*), same shape as the input.

        Returns:
            Scalar. If reduction is 'none', then (*), same shape as input.
        """
        # loss = focal_loss(
        #     input=input,
        #     target=target,
        #     alpha=self.alpha,
        #     gamma=self.gamma,
        #     reduction=self.reduction,
        # )
        loss = sigmoid_focal_loss(input, target, reduction="none")
        return loss

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)


@torch.jit.script
@dataclass
class PenaltyReducedFocalLoss:
    """Focal Loss class."""

    alpha: float
    gamma: int
    reduction: str

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Creates a criterion that computes Focal loss.

        Args:
            input: (*), where * means any number of dimensions.
            target: (*), same shape as the input.

        Returns:
            Scalar. If reduction is 'none', then (*), same shape as input.
        """
        loss = penalty_reduced_focal_loss(
            input=input,
            target=target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)
