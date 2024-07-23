"""Network decoder."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.functional import Tensor

from torchbox3d.math.conversions import BCHW_to_BKC
from torchbox3d.math.linalg.lie.SO3 import yaw_to_quat
from torchbox3d.math.ops.coding import decode_range_view
from torchbox3d.math.ops.nms import batched_multiclass_nms


@dataclass
class RangeDecoder:
    enable_azimuth_invariant_targets: bool
    enable_sample_by_range: bool

    lower_bounds: ListConfig
    upper_bounds: ListConfig
    subsampling_rates: ListConfig

    def decode(
        self,
        multiscale_outputs: Dict[Union[int, str], Dict[str, Tensor]],
        post_processing_config: DictConfig,
        task_config: DictConfig,
        use_nms: bool = True,
        **kwargs: Dict[int, Any],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decode the length, width, height cuboid parameterization."""
        predictions_list = []
        for k1, multiscale_outputs in multiscale_outputs.items():
            stride = cast(int, k1)
            cart = multiscale_outputs["cart"]
            mask = multiscale_outputs["mask"]

            task_offset = 0
            for k2, task_group in task_config.items():
                task_id = cast(int, k2)
                outputs = multiscale_outputs[task_id]

                classification_scores = outputs["logits"].sigmoid()
                scores = classification_scores * mask

                scores, categories = cast(
                    Tuple[Tensor, Tensor], scores.max(dim=1, keepdim=True)
                )

                cuboids = decode_range_view(
                    regressands=outputs["regressands"],
                    cart=cart,
                    enable_azimuth_invariant_targets=self.enable_azimuth_invariant_targets,
                )

                if self.enable_sample_by_range:
                    scores, categories, cuboids = sample_by_range(
                        scores,
                        categories,
                        cuboids,
                        cart,
                        tuple(self.lower_bounds),
                        tuple(self.upper_bounds),
                        tuple(self.subsampling_rates),
                    )
                else:
                    scores = BCHW_to_BKC(scores).squeeze(-1)
                    cuboids = BCHW_to_BKC(cuboids)
                    categories = (BCHW_to_BKC(categories)).squeeze(-1)

                categories += task_offset
                task_offset += len(task_group)

                predictions_list.append(
                    {
                        "scores": scores,
                        "cuboids": cuboids,
                        "categories": categories,
                    }
                )

        collated_predictions = defaultdict(list)
        for stride_predictions in predictions_list:
            for k, v in stride_predictions.items():
                collated_predictions[k].append(v)

        predictions = {k: torch.cat(v, dim=1) for k, v in collated_predictions.items()}

        params = predictions["cuboids"]
        scores = predictions["scores"]
        categories = predictions["categories"]
        if use_nms:
            params, scores, categories, batch_index = batched_multiclass_nms(
                params,
                scores,
                categories,
                num_pre_nms=post_processing_config["num_pre_nms"],
                num_post_nms=post_processing_config["num_post_nms"],
                iou_threshold=post_processing_config["nms_threshold"],
                min_confidence=post_processing_config["min_confidence"],
                nms_mode=post_processing_config["nms_mode"],
            )
        else:
            B, N, _ = params.shape
            batch_index = torch.arange(0, B, device=params.device).repeat_interleave(N)
            params = params.flatten(0, 1)
            scores = scores.flatten(0, 1)
            categories = categories.flatten(0, 1)

            t = scores >= post_processing_config["min_confidence"]
            params = params[t]
            scores = scores[t]
            categories = categories[t]
            batch_index = batch_index[t]

        quats_wxyz = yaw_to_quat(params[:, -1:])
        params = torch.cat([params[:, :-1], quats_wxyz], dim=-1)
        return params, scores, categories, batch_index


def sample_by_range(
    scores: Tensor,
    categories: Tensor,
    cuboids: Tensor,
    cart: Tensor,
    lower_bounds: Tuple[float, ...],
    upper_bounds: Tuple[float, ...],
    subsampling_rates: Tuple[int, ...],
) -> Tuple[Tensor, Tensor, Tensor]:
    scores_list = []
    categories_list = []
    cuboids_list = []

    dists = cart.norm(dim=1, keepdim=True)
    lower_bounds = torch.as_tensor(lower_bounds, device=scores.device).view(1, -1, 1, 1)
    upper_bounds = torch.as_tensor(upper_bounds, device=scores.device).view(1, -1, 1, 1)
    partitions = torch.logical_and(dists > lower_bounds, dists <= upper_bounds)
    for i, partition in enumerate(partitions.transpose(1, 0)):
        rate = subsampling_rates[i]
        scores_list.append(
            ((scores * partition.unsqueeze(1))[:, :, ::, ::rate]).flatten(2)
        )
        categories_list.append(categories[:, :, ::, ::rate].flatten(2))
        cuboids_list.append(cuboids[:, :, ::, ::rate].flatten(2))

    scores = torch.cat(scores_list, dim=-1)
    categories = torch.cat(categories_list, dim=-1)
    cuboids = torch.cat(cuboids_list, dim=-1)

    return scores.squeeze(1), categories.squeeze(1), cuboids.transpose(2, 1)
