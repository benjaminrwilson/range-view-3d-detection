from typing import Final, Tuple

import torch
from mmcv.ops import box_iou_rotated
from torch import Tensor

XYLWA_INDICES: Final = (0, 1, 3, 4, 6)
LW_INDICES: Final = (3, 4)


def iou_3d_axis_aligned(cuboids_a: Tensor, cuboids_b: Tensor) -> Tuple[Tensor, Tensor]:
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
    if not object_ious.isfinite().all():
        mask = object_ious.isfinite().logical_not()
        error_a = cuboids_a[mask]
        invalid_ious = object_ious[mask]
        logger.error(f"Cuboids A: {error_a}.")
        logger.error(f"Invalid IoUs: {invalid_ious}.")
        raise RuntimeError("Invalid IoUs.")
    return object_ious, iou_bev


def iou(src_dims_m: Tensor, target_dims_m: Tensor) -> Tensor:
    """Compute aligned IoU."""
    inter = torch.minimum(src_dims_m, target_dims_m).prod(axis=1, keepdim=True)
    union = torch.maximum(src_dims_m, target_dims_m).prod(axis=1, keepdim=True)
    iou = torch.divide(inter, union)
    return iou
