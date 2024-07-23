"""Functional interface."""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def varifocal_loss(
    input: Tensor,
    target: Tensor,
    alpha: float,
    gamma: float,
    reduction: str,
) -> Tensor:
    """Compute Varifocal loss."""
    bce_loss = F.binary_cross_entropy_with_logits(
        input=input, target=target, reduction=reduction
    )
    likelihoods = input.sigmoid()
    foreground_mask = target > 0.0
    background_mask = target == 0

    foreground_loss = foreground_mask * target * bce_loss
    background_loss = alpha * background_mask * likelihoods.pow(gamma) * bce_loss
    loss = foreground_loss + background_loss
    return loss


@torch.jit.script
def penalty_reduced_focal_loss(
    input: Tensor,
    target: Tensor,
    alpha: float,
    gamma: int,
    reduction: str,
) -> Tensor:
    """Compute focal loss."""
    bce_loss = F.binary_cross_entropy_with_logits(
        input=input, target=target, reduction=reduction
    )
    likelihoods = input.sigmoid()
    foreground_mask = target == 1
    background_mask = (1 - target).pow(4.0)

    foreground_loss = foreground_mask * (1 - likelihoods).pow(gamma) * bce_loss
    background_loss = alpha * background_mask * likelihoods.pow(gamma) * bce_loss
    loss = foreground_loss + background_loss
    return loss
