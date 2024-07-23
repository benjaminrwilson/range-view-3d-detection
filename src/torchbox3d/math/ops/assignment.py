"""Foreground assignment ops."""

import logging
from typing import Final, Tuple, cast

import torch
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated
from omegaconf import DictConfig
from torch import Tensor

from torchbox3d.math.ops.coding import decode_range_view

XYLWA_INDICES: Final = (0, 1, 3, 4, 6)
LW_INDICES: Final = (3, 4)

logger = logging.getLogger(__name__)


def iou_3d_axis_aligned(cuboids_a: Tensor, cuboids_b: Tensor, **kwargs: int) -> Tensor:
    """Compute axis-aligned 3D IoU."""
    cuboids_a_bev = cuboids_a[:, XYLWA_INDICES].contiguous()
    cuboids_b_bev = cuboids_b[:, XYLWA_INDICES].contiguous()
    iou_bev = box_iou_rotated(
        cuboids_a_bev.float(), cuboids_b_bev.float(), aligned=True
    ).clamp(0.0, 1.0)
    iou_bev = iou_bev.nan_to_num(nan=0.0)

    areas_a = cuboids_a[:, LW_INDICES].prod(dim=-1)
    areas_b = cuboids_b[:, LW_INDICES].prod(dim=-1)

    overlaps_bev = iou_bev * (areas_a + areas_b) / (1.0 + iou_bev)

    pds_top = cuboids_a[:, 2] + cuboids_a[:, 5] / 2.0
    pds_btm = cuboids_a[:, 2] - cuboids_a[:, 5] / 2.0

    gts_top = cuboids_b[:, 2] + cuboids_b[:, 5] / 2.0
    gts_btm = cuboids_b[:, 2] - cuboids_b[:, 5] / 2.0

    highest_of_btm = torch.max(pds_btm, gts_btm)
    lowest_of_top = torch.min(pds_top, gts_top)
    overlaps_h = torch.clamp(lowest_of_top - highest_of_btm, min=0)
    overlaps_3d = overlaps_bev * overlaps_h

    volume1 = cuboids_a[:, 3:6].prod(dim=-1)
    volume2 = cuboids_b[:, 3:6].prod(dim=-1)
    object_ious = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)
    object_ious = object_ious.nan_to_num(nan=0.0)

    if kwargs["normalize_affinities"]:
        object_ious /= object_ious.max() + 1e-8

    # Sanity check.
    if not object_ious.isfinite().all():
        mask = object_ious.isfinite().logical_not()
        error_a = cuboids_a[mask]
        invalid_ious = object_ious[mask]
        logger.error(f"Cuboids A: {error_a}.")
        logger.error(f"Invalid IoUs: {invalid_ious}.")
        raise RuntimeError("Invalid IoUs.")
    return object_ious


def iou_2d_axis_aligned(cuboids_a: Tensor, cuboids_b: Tensor, **kwargs: int) -> Tensor:
    """Compute axis-aligned 3D IoU."""
    cuboids_a_bev = cuboids_a[:, XYLWA_INDICES].contiguous()
    cuboids_b_bev = cuboids_b[:, XYLWA_INDICES].contiguous()
    iou_bev = box_iou_rotated(
        cuboids_a_bev.float(), cuboids_b_bev.float(), aligned=True
    ).clamp(0.0, 1.0)
    if kwargs["normalize_affinities"]:
        object_ious /= object_ious.max() + 1e-8
    return iou_bev


def compute_classification_targets(
    input: Tensor,
    target: Tensor,
    classification_labels: Tensor,
    cart: Tensor,
    targets_config: DictConfig,
    mask: Tensor,
    panoptics: Tensor,
    background_index: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute classification targets."""
    # Important! Block gradient flow.
    input_detached = input.detach()

    all_foreground = (
        F.one_hot(classification_labels, background_index + 1)
        .permute(0, 3, 1, 2)[:, :-1]
        .float()
    )

    name = cast(str, targets_config.affinity_fn).upper()
    if name == "BEV":
        affinity_fn = iou_2d_axis_aligned
    elif name == "GAUSSIAN":
        affinity_fn = _gaussian
    else:
        raise NotImplementedError("This affinity function is not implemented.")

    # Decode inputs and targets.
    pds = decode_range_view(
        input_detached,
        cart=cart,
        enable_azimuth_invariant_targets=True,
    ).squeeze(1)
    gts = decode_range_view(
        target,
        cart=cart,
        enable_azimuth_invariant_targets=targets_config.enable_azimuth_invariant_targets,
    ).squeeze(1)

    affinities = torch.zeros_like(target[:, 0:1])
    foreground_mask = torch.zeros_like(target[:, 0:1])
    for i, _ in enumerate(panoptics):
        panoptic_mask = F.one_hot(panoptics[i]).permute(0, 3, 1, 2)[:, 1:].squeeze(0)
        for _, instance_mask in enumerate(panoptic_mask):
            instance_mask = instance_mask.bool()
            if instance_mask.sum() == 0:
                continue

            dts_i = pds[i : i + 1].masked_select(instance_mask).view(7, -1).t()
            gts_i = gts[i : i + 1].masked_select(instance_mask).view(7, -1).t()

            affinities_i = affinity_fn(dts_i, gts_i, **dict(targets_config))
            k_actual = min(targets_config.k, len(affinities_i))

            likelihoods, indices = affinities_i.topk(k=k_actual)
            likelihoods = torch.zeros_like(affinities_i).scatter(
                0, indices, likelihoods
            )
            affinities[i : i + 1].masked_scatter_(
                instance_mask, likelihoods.type_as(affinities)
            )
            foreground_mask[i : i + 1].masked_scatter_(
                instance_mask, likelihoods.bool().type_as(affinities)
            )

    background_mask = torch.logical_and(foreground_mask.logical_not(), mask)
    affinities = affinities * all_foreground
    regression_weights = all_foreground.any(dim=1, keepdim=True)
    return (
        affinities,
        foreground_mask,
        background_mask,
        regression_weights,
    )


def _gaussian(cuboids_a: Tensor, cuboids_b: Tensor, **kwargs: int) -> Tensor:
    dists = cast(
        Tensor,
        torch.linalg.norm(cuboids_a[:, :3] - cuboids_b[:, :3], dim=-1),
    )
    if kwargs["normalize_affinities"]:
        dists -= dists.min()
    likelihoods = torch.exp(-dists / kwargs["sigma"] ** 2)
    return likelihoods


def normalize_instance_affinities(
    affinities: Tensor, category_labels: Tensor, panoptics: Tensor
) -> Tensor:
    affinities = affinities.sum(dim=1, keepdim=True)
    for i, _ in enumerate(panoptics):
        panoptic_mask = F.one_hot(panoptics[i]).permute(0, 3, 1, 2)[:, 1:].squeeze(0)
        for instance_mask in panoptic_mask:
            if instance_mask.sum() == 0:
                continue

            affinities_i = affinities[i : i + 1].masked_select(instance_mask.bool())
            max_i = affinities_i.max()
            if max_i > 0:
                affinities_i /= max_i

            affinities[i : i + 1].masked_scatter_(instance_mask.bool(), affinities_i)
    return affinities * category_labels
