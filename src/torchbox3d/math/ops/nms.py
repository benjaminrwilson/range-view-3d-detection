from typing import List, Tuple

import torch
import torch.nn.functional as F
import weighted_nms_ext
from detectron2.layers.nms import nms_rotated
from torch import Tensor


# @torch.jit.script
def hard_multiclass_nms(
    cuboids_i: Tensor,
    scores_i: Tensor,
    categories_i: Tensor,
    iou_threshold: float,
    num_pre_nms: int,
    num_post_nms: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    cuboids_list: List[Tensor] = []
    scores_list: List[Tensor] = []
    categories_list: List[Tensor] = []
    unique_categories: Tensor = torch.unique(categories_i)
    for j in unique_categories:
        category_mask = categories_i == j
        scores_ij = scores_i[category_mask]
        cuboids_ij = cuboids_i[category_mask]
        categories_ij = categories_i[category_mask]

        k = min(len(scores_ij), num_pre_nms)
        scores_ij, rank_ij = scores_ij.topk(k=k, dim=0)
        categories_ij = categories_ij[rank_ij]
        cuboids_ij = cuboids_ij[rank_ij]
        input = cuboids_ij[:, [0, 1, 3, 4, 6]].contiguous()
        # _, keep_ij = mmcv_nms_rotated(
        #     dets=input,
        #     scores=scores_ij,
        #     iou_threshold=iou_threshold,
        # )

        input[:, -1] = -input[:, -1].rad2deg()
        keep_ij = nms_rotated(
            boxes=input.type(torch.float32),
            scores=scores_ij.type(torch.float32),
            iou_threshold=torch.as_tensor(iou_threshold),
        )

        cuboids_ij = cuboids_ij[keep_ij]
        scores_ij = scores_ij[keep_ij]

        scores_ij = scores_ij.flatten()
        categories_ij = torch.full_like(scores_ij, fill_value=j)

        k = min(len(cuboids_ij), num_post_nms)
        scores_ij, rank_ij = scores_ij.topk(k=k, dim=0)
        categories_ij = categories_ij[rank_ij]
        cuboids_ij = cuboids_ij[rank_ij]

        cuboids_list.append(cuboids_ij)
        scores_list.append(scores_ij)
        categories_list.append(categories_ij)
    return torch.cat(cuboids_list), torch.cat(scores_list), torch.cat(categories_list)


def weighted_multiclass_nms(
    cuboids_i: Tensor,
    scores_i: Tensor,
    categories_i: Tensor,
    iou_threshold: float,
    num_pre_nms: int,
    num_post_nms: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Multi-class weighted non-maximum-suppression."""
    cuboids_list: List[Tensor] = []
    scores_list: List[Tensor] = []
    categories_list: List[Tensor] = []
    unique_categories: Tensor = categories_i.unique()
    for j in unique_categories:
        category_mask = categories_i == j
        scores_ij = scores_i[category_mask]
        cuboids_ij = cuboids_i[category_mask]
        categories_ij = categories_i[category_mask]

        k = min(len(scores_ij), num_pre_nms)
        scores_ij, rank_ij = scores_ij.topk(k=k, dim=0)
        categories_ij = categories_ij[rank_ij]
        cuboids_ij = cuboids_ij[rank_ij]
        boxes_ij = cuboids_ij[..., [0, 1, 3, 4, 6]].contiguous()
        input = torch.concatenate(
            [
                boxes_ij[:, :2] - boxes_ij[:, 2:4] / 2,
                boxes_ij[:, :2] + boxes_ij[:, 2:4] / 2,
                boxes_ij[:, -1:],
            ],
            dim=-1,
        )

        sin = cuboids_ij[:, -1:].sin()
        cos = cuboids_ij[:, -1:].cos()

        cuboids_ij = torch.concatenate([cuboids_ij[:, :-1], sin, cos], dim=1)
        _, cuboids_ij, _ = weighted_nms(
            input,
            cuboids_ij,
            scores_ij,
            nms_threshold=iou_threshold,
            merge_thresh=0.5,
        )

        cuboids_ij, sin_ij, cos_ij, scores_ij = cuboids_ij.split([6, 1, 1, 1], dim=1)
        az_ij = torch.atan2(sin_ij, cos_ij)
        cuboids_ij = torch.concatenate([cuboids_ij, az_ij], dim=1)
        scores_ij = scores_ij.flatten()
        categories_ij = torch.full_like(scores_ij, fill_value=j)

        k = min(len(cuboids_ij), num_post_nms)
        scores_ij, rank_ij = scores_ij.topk(k=k, dim=0)
        categories_ij = categories_ij[rank_ij]
        cuboids_ij = cuboids_ij[rank_ij]

        cuboids_list.append(cuboids_ij)
        scores_list.append(scores_ij)
        categories_list.append(categories_ij)
    return torch.cat(cuboids_list), torch.cat(scores_list), torch.cat(categories_list)


def weighted_nms(
    boxes: Tensor,
    data2merge: Tensor,
    scores: Tensor,
    nms_threshold: float,
    merge_thresh: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # """Weighted NMS function with gpu implementation.
    #    Modification from the cpu version in https://github.com/TuSimple/RangeDet

    # Args:
    #     boxes (torch.Tensor): Input boxes with the shape of [N, 5]
    #         ([x1, y1, x2, y2, ry]).
    #     data2merge (torch.Tensor): Input data with the shape of [N, C], corresponding to boxes.
    #         If you want to merge origin boxes, just let data2merge == boxes
    #     scores (torch.Tensor): Scores of boxes with the shape of [N].
    #     thresh (float): Threshold.
    #     merge_thresh (float): boxes have IoUs with the current box higher than the threshold with weighted merged by scores.

    # Returns:
    #     torch.Tensor: Indexes after nms.
    # """
    sorted_scores, order = scores.sort(0, descending=True)

    boxes = boxes[order].contiguous().float()
    data2merge = data2merge[order].contiguous().float()
    data2merge_score = (
        torch.cat([data2merge, sorted_scores[:, None]], 1).contiguous().float()
    )
    output = torch.zeros_like(data2merge_score)
    count = torch.zeros(boxes.size(0), dtype=torch.long, device=boxes.device)

    assert data2merge_score.dim() == 2

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = weighted_nms_ext.wnms_gpu(
        boxes,
        data2merge_score,
        output,
        keep,
        count,
        nms_threshold,
        merge_thresh,
        boxes.device.index,
    )
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()

    assert output[num_out:, :].sum() == 0
    assert (count[:num_out] > 0).all()
    count = count[:num_out]
    output = output[:num_out, :]
    return keep, output, count


# @torch.jit.script
def batched_multiclass_nms(
    cuboids: Tensor,
    scores: Tensor,
    categories: Tensor,
    num_pre_nms: int,
    num_post_nms: int,
    iou_threshold: float,
    min_confidence: float,
    nms_mode: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply a variant of non-maximum-suppression to a batched set of proposals.

    Notation:
        B: Number of batches.
        C: Number of classes.
        H: Spatial height.
        W: Spatial width.

    Takes a BCHW tensor of proposals and clusters them into a set of K proposals where K <= B*C*H*W.
    """
    num_batches, _, num_parameters = cuboids.shape
    cuboids_list: List[Tensor] = []
    scores_list: List[Tensor] = []
    categories_list: List[Tensor] = []
    batch_index_list: List[Tensor] = []

    nms_mode = nms_mode.upper()

    # Iterate over all batches since they're mutually exclusive.
    for i in range(num_batches):
        # Filter by minimum confidence.
        min_confidence_mask = scores[i] >= min_confidence
        cuboids_i = cuboids[i, min_confidence_mask]
        scores_i = scores[i, min_confidence_mask]
        categories_i = categories[i, min_confidence_mask]

        no_detections_exist = scores_i.shape[0] == 0
        if no_detections_exist:
            continue

        if nms_mode == "HARD":
            cuboids_i, scores_i, categories_i = hard_multiclass_nms(
                cuboids_i=cuboids_i,
                scores_i=scores_i,
                categories_i=categories_i,
                iou_threshold=iou_threshold,
                num_pre_nms=num_pre_nms,
                num_post_nms=num_post_nms,
            )
        elif nms_mode == "WEIGHTED":
            cuboids_i, scores_i, categories_i = weighted_multiclass_nms(
                cuboids_i=cuboids_i,
                scores_i=scores_i,
                categories_i=categories_i,
                iou_threshold=iou_threshold,
                num_pre_nms=num_pre_nms,
                num_post_nms=num_post_nms,
            )
        else:
            raise NotImplementedError(f"NMS Mode: {nms_mode} is not implemented.")

        batch_index = torch.full_like(scores_i, fill_value=i)

        cuboids_list.append(cuboids_i)
        scores_list.append(scores_i)
        categories_list.append(categories_i)
        batch_index_list.append(batch_index)

    # Initialize to empty tensors.
    suppressed_cuboids = cuboids.new_empty(size=(0, num_parameters))
    suppressed_scores = scores.new_empty(size=(0, 1))
    suppressed_categories = categories.new_empty(size=(0, 1))
    suppressed_batch_index = categories.new_empty(size=(0, 1))

    if len(cuboids_list) > 0:
        suppressed_cuboids = torch.cat(cuboids_list)
        suppressed_scores = torch.cat(scores_list)
        suppressed_categories = torch.cat(categories_list)
        suppressed_batch_index = torch.cat(batch_index_list)

    return (
        suppressed_cuboids,
        suppressed_scores,
        suppressed_categories,
        suppressed_batch_index,
    )
