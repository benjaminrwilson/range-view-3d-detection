"""Network head."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Final, List, Tuple, Union, cast

import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, ListConfig
from pytorch_lightning.core.module import LightningModule
from torch import nn
from torch.functional import Tensor
from torch.utils.data import default_collate

from torchbox3d.math.ops.assignment import compute_classification_targets
from torchbox3d.math.polytope import compute_interior_points_mask, cuboids_to_vertices
from torchbox3d.nn.heads.dense_head import DenseHead
from torchbox3d.utils.polars import polars_to_torch

COLS: Final = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "task_id",
    "offset",
    "batch_index",
)

FOCAL_PRIOR_PROB: Final = 0.01


@dataclass(unsafe_hash=True)
class DetectionHead(LightningModule):
    """DetectionHead class for keypoint classification and regression."""

    fpn: DictConfig
    fpn_kernel_sizes: DictConfig
    targets_config: DictConfig

    num_classification_blocks: int
    num_regression_blocks: int

    final_kernel_size: int
    tasks_cfg: DictConfig
    task_in_channels: int

    classification_weight: float
    regression_weight: float
    coding_weights: ListConfig

    classification_head_channels: int
    regression_head_channels: int

    classification_normalization_method: str

    additive_smoothing: float = 1.0

    _cls_loss: DictConfig = MISSING
    _regression_loss: DictConfig = MISSING

    classification_head: nn.ModuleDict = field(init=False)
    regression_head: nn.ModuleDict = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.classification_head = nn.ModuleDict(
            {
                str(stride): nn.ModuleDict(
                    {
                        str(k): DenseHead(
                            num_channels,
                            self.classification_head_channels,
                            len(categories),
                            kernel_size=self.fpn_kernel_sizes[stride],
                            final_kernel_size=self.final_kernel_size,
                            prior_prob=FOCAL_PRIOR_PROB,
                            num_blocks=self.num_classification_blocks,
                        )
                        for k, categories in self.tasks_cfg.items()
                    }
                )
                for stride, num_channels in self.fpn.items()
            }
        )

        self.regression_head = nn.ModuleDict(
            {
                str(stride): nn.ModuleDict(
                    {
                        str(k): DenseHead(
                            num_channels,
                            self.regression_head_channels,
                            8,
                            kernel_size=self.fpn_kernel_sizes[stride],
                            final_kernel_size=self.final_kernel_size,
                            num_blocks=self.num_regression_blocks,
                        )
                        for k, _ in self.tasks_cfg.items()
                    }
                )
                for stride, num_channels in self.fpn.items()
            }
        )
        self.cls_loss = instantiate(self._cls_loss)
        self.regression_loss = instantiate(self._regression_loss)

    def forward(
        self,
        input: Tensor,
        data: Dict[int, Dict[str, Tensor]],
        return_loss: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """Network forward pass."""
        multiscale_outputs: Dict[int, Dict[Union[str, int], Dict[int, Tensor]]] = {}

        for stride in self.fpn.keys():
            stride = cast(int, stride)
            stride_h, stride_w = 1, int(stride)
            multiscale_features = input[int(stride)]

            stride_h = int(stride_h)
            stride_w = int(stride_w)
            features = data["features"][:, :, ::stride_h, ::stride_w].clone()
            cart = data["cart"][:, :, ::stride_h, ::stride_w].clone()
            mask = data["mask"][:, :, ::stride_h, ::stride_w].clone()

            multiscale_outputs[int(stride)] = {
                "features": features,
                "cart": cart,
                "mask": mask,
            }

            if self.targets_config.fpn_assignment_method == "RANGE":
                dists = cart.norm(dim=1, keepdim=True)
                lower, upper = self.targets_config.range_partitions[stride]
                partition_mask = torch.logical_and(dists > lower, dists <= upper)
                mask *= partition_mask

            strided_classification_head = self.classification_head[str(stride)]
            strided_regression_head = self.regression_head[str(stride)]
            for task_id, _ in self.tasks_cfg.items():
                task_id = cast(int, task_id)
                logits = cast(
                    Tensor,
                    strided_classification_head[str(task_id)](
                        multiscale_features,
                        cart,
                        mask,
                    ),
                )
                regressands = cast(
                    Tensor,
                    strided_regression_head[str(task_id)](
                        multiscale_features,
                        cart,
                        mask,
                    ),
                )

                multiscale_outputs[stride][task_id] = {
                    "logits": logits,
                    "regressands": regressands,
                }

        losses: Dict[str, Any] = {}
        if return_loss:
            targets = compute_targets(
                data,
                tasks_config=self.tasks_cfg,
                fpn_strides=self.fpn.keys(),
                targets_config=self.targets_config,
            )
            for k, v in targets.items():
                data[k] = v
            losses = self.loss(multiscale_outputs, data)
        return multiscale_outputs, losses

    def loss(
        self,
        multiscale_outputs: Dict[
            int, Dict[Union[int, str], Union[Tensor, Dict[str, Tensor]]]
        ],
        multiscale_data: Dict[int, Dict[int, Dict[str, Tensor]]],
    ) -> Dict[str, Any]:
        """Compute the classification and regression losses."""
        multiscale_losses: Dict[int, Dict[int, Dict[str, Tensor]]] = {}

        # For all strides ...
        for stride in self.fpn.keys():
            stride = cast(int, stride)
            multiscale_losses[stride] = {}
            mask = cast(Tensor, multiscale_outputs[stride]["mask"])

            for key, _ in self.tasks_cfg.items():
                task_id = cast(int, key)
                # Compute targets, foreground, and background.
                (
                    targets,
                    task_fg_mask,
                    task_bg_mask,
                    task_reg_mask,
                ) = self.compute_classification_targets(
                    multiscale_outputs=cast(
                        Dict[Union[int, str], Dict[str, Tensor]],
                        multiscale_outputs[stride],
                    ),
                    multiscale_data=multiscale_data[stride],
                    task_id=task_id,
                )

                # Compute classification loss, foreground loss, and background loss.
                classification_loss = self.compute_classification_loss(
                    inputs=cast(
                        Dict[Union[int, str], Dict[str, Tensor]],
                        multiscale_outputs[stride],
                    ),
                    data=multiscale_data[stride],
                    targets=targets,
                    task_id=task_id,
                )

                # Compute regression loss.
                regression_loss = self.compute_regression_loss(
                    task_regression_mask=task_reg_mask,
                    coding_weights=list(self.coding_weights),
                    inputs=cast(
                        Dict[Union[int, str], Dict[str, Tensor]],
                        multiscale_outputs[stride],
                    ),
                    data=multiscale_data[stride],
                    task_id=task_id,
                )

                strided_data = multiscale_data[stride]
                strided_object_point_counts = strided_data[task_id]["points_per_obj"]

                multiscale_losses[stride][task_id] = {
                    "classification_loss": classification_loss,
                    "regression_loss": regression_loss,
                    "targets": targets,
                    "foreground": task_fg_mask.float(),
                    "background": task_bg_mask.float(),
                    "mask": mask,
                    "point_counts": strided_object_point_counts,
                }

        return reduce_multiscale_loss(
            multiscale_losses=multiscale_losses,
            multiscale_data=multiscale_data,
            additive_smoothing=self.additive_smoothing,
        )

    def compute_classification_targets(
        self,
        multiscale_outputs: Dict[Union[int, str], Dict[str, Tensor]],
        multiscale_data: Dict[int, Dict[str, Tensor]],
        task_id: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Dynamically compute targets from network outputs."""
        cart = cast(Tensor, multiscale_outputs["cart"])
        mask = cast(Tensor, multiscale_outputs["mask"])

        input = multiscale_outputs[task_id]["regressands"]
        target = multiscale_data[task_id]["regression_targets"]
        classification_labels = multiscale_data[task_id]["classification_labels"]
        panoptics = multiscale_data[task_id]["panoptics"]

        background_index = len(self.tasks_cfg[task_id])
        (
            targets,
            foreground_mask,
            background_mask,
            regression_weights,
        ) = compute_classification_targets(
            input=input,
            target=target,
            classification_labels=classification_labels,
            cart=cart,
            targets_config=self.targets_config,
            mask=mask,
            panoptics=panoptics,
            background_index=background_index,
        )
        return (
            targets,
            foreground_mask,
            background_mask,
            regression_weights,
        )

    def compute_classification_loss(
        self,
        inputs: Dict[Union[int, str], Dict[str, Tensor]],
        data: Dict[int, Dict[str, Tensor]],
        targets: Tensor,
        task_id: int,
    ) -> Tensor:
        """Compute classification loss."""
        logits = inputs[task_id]["logits"]
        mask = cast(Tensor, inputs["mask"])
        classification_loss = (
            self.classification_weight * self.cls_loss(logits, targets) * mask
        )

        data[task_id]["targets"] = targets
        return cast(Tensor, classification_loss)

    def compute_regression_loss(
        self,
        task_regression_mask: Tensor,
        coding_weights: List[float],
        inputs: Dict[Union[int, str], Dict[str, Tensor]],
        data: Dict[int, Dict[str, Tensor]],
        task_id: int,
    ) -> Tensor:
        """Compute regression loss."""
        task_inputs = inputs[task_id]
        task_data = data[task_id]
        mask = inputs["mask"]

        regressands = task_inputs["regressands"]
        regression_targets = task_data["regression_targets"]

        coding_weights_tch = regressands.new(coding_weights).view(1, -1, 1, 1)
        task_regression_normalization = (
            (data[task_id]["points_per_obj"] + self.additive_smoothing)
            .double()
            .reciprocal()
        )

        regression_loss = (
            cast(
                Tensor,
                self.regression_loss(regressands, regression_targets),
            )
            * self.regression_weight
            * task_regression_mask
            * task_regression_normalization
            * mask
            * coding_weights_tch
            / coding_weights_tch.shape[1]
        )
        return regression_loss


def reduce_multiscale_loss(
    multiscale_losses: Dict[int, Dict[int, Dict[str, Tensor]]],
    multiscale_data: Dict[int, Dict[int, Dict[str, Tensor]]],
    additive_smoothing: float,
) -> Dict[str, Any]:
    """Compute multi-scale loss."""
    losses_list: List[Dict[str, Tensor]] = []
    auxillary: DefaultDict[str, Dict[int, Dict[str, Tensor]]] = defaultdict(dict)

    # Compute total objects across all tasks.
    num_object_list = []
    for key, data in multiscale_data.items():
        if not isinstance(key, int):
            continue
        for _, values in data.items():
            panoptics = values["panoptics"]
            num_objects = torch.as_tensor(
                [x.unique()[1:].shape[0] for x in panoptics], device=panoptics.device
            ).sum()
            num_object_list.append(num_objects)

    total_objects = torch.stack(num_object_list).sum().clamp(1.0)

    # Compute total foreground across all tasks.
    fg_list = []
    for key, data in multiscale_losses.items():
        for _, values in data.items():
            num_fg = values["foreground"].sum()
            fg_list.append(num_fg)
    total_fg = torch.stack(fg_list).sum() + additive_smoothing

    for key, data in multiscale_losses.items():
        auxillary["aux"][key] = {}
        for task_id, task_losses in data.items():
            if not isinstance(task_id, int):
                continue
            cls_loss = task_losses["classification_loss"] / total_fg
            fg_loss = torch.sum(cls_loss * task_losses["foreground"])
            bg_loss = torch.sum(cls_loss * task_losses["background"])
            cls_loss = cls_loss.sum()

            regression_loss = task_losses["regression_loss"] / total_objects
            coordinate_loss, dimension_loss, rotation_loss = (
                regression_loss.sum(dim=[2, 3]).sum(dim=0).split([3, 3, 2], dim=-1)
            )

            coordinate_loss = cast(Tensor, coordinate_loss.sum())
            dimension_loss = cast(Tensor, dimension_loss.sum())
            rotation_loss = cast(Tensor, rotation_loss.sum())
            regression_loss = coordinate_loss + dimension_loss + rotation_loss
            loss = cls_loss + regression_loss

            auxillary["aux"][key][task_id] = {
                k: v.detach() for k, v in task_losses.items()
            }
            task_losses = {
                "loss": loss,
                "classification_loss": cls_loss.detach(),
                "foreground_loss": fg_loss.detach(),
                "background_loss": bg_loss.detach(),
                "regression_loss": regression_loss.detach(),
                "coordinate_loss": coordinate_loss.detach(),
                "dimension_loss": dimension_loss.detach(),
                "rotation_loss": rotation_loss.detach(),
                "total_fg": total_fg,
                "total_objects": total_objects,
            }
            losses_list.append(task_losses)

    data = cast(Dict[str, Tensor], default_collate(losses_list))
    losses = {k: v.sum() for k, v in data.items()}

    # Add strided losses for debugging.
    strides = list(multiscale_losses.keys())
    for k, v in data.items():
        for i, key in enumerate(strides):
            name = f"{k}/s{key}"
            losses[name] = v[i]
    losses |= auxillary
    return losses


@torch.jit.script
def rotate(offset: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    x1 = cos * offset[:, :, 0] + sin * offset[:, :, 1]
    x2 = -sin * offset[:, :, 0] + cos * offset[:, :, 1]
    x3 = offset[:, :, 2]
    return torch.stack((x1, x2, x3), dim=-1)


@torch.jit.script
def encode_regression_targets(
    cuboids: Tensor,
    interior_points: Tensor,
    enable_azimuth_invariant_targets: bool,
) -> Tensor:
    """Encode the network regression targets."""
    targets = interior_points.new_zeros((cuboids.shape[0], interior_points.shape[0], 8))
    offset = cuboids[:, None, :3].float() - interior_points
    rots = cuboids[:, None, 6:7]
    if enable_azimuth_invariant_targets:
        azimuth_points = torch.atan2(interior_points[:, 1:2], interior_points[:, 0:1])
        rots = rots - azimuth_points

        cos = torch.cos(azimuth_points).squeeze(1)
        sin = torch.sin(azimuth_points).squeeze(1)

        mats = torch.eye(3, device=cuboids.device, dtype=offset.dtype)[
            None
        ].repeat_interleave(
            len(azimuth_points),
            dim=0,
        )
        mats[:, 0, 0] = cos
        mats[:, 0, 1] = sin
        mats[:, 1, 0] = -sin
        mats[:, 1, 1] = cos
        offset = rotate(offset, sin, cos)

    targets[:, :, :3] = offset
    targets[:, :, 3:6] = cuboids[:, None, 3:6].log()
    targets[:, :, 6:7] = torch.sin(rots)
    targets[:, :, 7:8] = torch.cos(rots)
    return targets


def compute_targets(
    x: Dict[str, Union[pl.DataFrame, Tensor]],
    tasks_config: DictConfig,
    fpn_strides: List[int],
    targets_config: DictConfig,
) -> Dict[str, Tensor]:
    range_partitions = targets_config.range_partitions
    fpn_assignment_method = targets_config.fpn_assignment_method
    enable_azimuth_invariant_targets = targets_config.enable_azimuth_invariant_targets

    cart = cast(Tensor, x["cart"])
    annotations = polars_to_torch(x["annotations"], columns=COLS, device=cart.device)

    _, _, H, W = cart.shape
    vertices = cuboids_to_vertices(
        annotations[None, :, :7].float().contiguous()
    ).squeeze(0)

    batch_indices, counts = annotations[:, -1].unique(return_counts=True)
    batch_vertices = cast(List[Tensor], vertices.split(counts.tolist(), dim=0))
    batch_annotations = cast(List[Tensor], annotations.split(counts.tolist(), dim=0))
    batch_task_ids = cast(
        List[Tensor], annotations[:, -3].long().split(counts.tolist(), dim=0)
    )

    # Stride -> tasks -> targets.
    tgts = initialize_targets(
        tasks_cfg=tasks_config,
        fpn_strides=fpn_strides,
        shape=cart.shape,
        device=cart.device,
    )

    batch_indices = batch_indices.long().tolist()
    for i, batch_index in enumerate(batch_indices):
        cart_i = cart[batch_index]
        verts_i = batch_vertices[i]

        mask_i = compute_interior_points_mask(
            cart_i.flatten(1, 2).t().contiguous().double(), verts_i.double()
        ).view(-1, H, W)

        full_resolution_interior_pts = mask_i.flatten(1, 2).sum(dim=-1)
        for _, stride in enumerate(fpn_strides):
            stride_h, stride_w = (1, stride)
            strided_width = int(W / stride_w)
            strided_height = int(H / stride_h)

            cart_ij = cart_i[:, ::stride_h, ::stride_w].flatten(1, 2).t()
            mask_ij = mask_i[:, ::stride_h, ::stride_w].flatten(1, 2)

            annotations_ij = batch_annotations[i].clone()
            ids_ij = batch_task_ids[i].clone()
            full_resolution_interior_pts_ij = full_resolution_interior_pts.clone()

            if fpn_assignment_method == "RANGE":
                dists_ij = annotations_ij[:, :3].norm(dim=-1)
                lower, upper = range_partitions[stride]
                partition_mask = torch.logical_and(dists_ij > lower, dists_ij <= upper)

                annotations_ij = annotations_ij[partition_mask]
                dists_ij = dists_ij[partition_mask]
                mask_ij = mask_ij[partition_mask]
                ids_ij = ids_ij[partition_mask]
                full_resolution_interior_pts_ij = full_resolution_interior_pts_ij[
                    partition_mask
                ]
                # No annotations within range partition.
                if dists_ij.shape[0] == 0:
                    continue

            task_indices, task_counts = ids_ij.unique(return_counts=True)
            task_annotations_list = annotations_ij.split(task_counts.tolist())
            task_mask_list = mask_ij.split(task_counts.tolist())
            task_full_resolution_interior_pts_list = (
                full_resolution_interior_pts_ij.split(task_counts.tolist())
            )
            for k, t_id in enumerate(task_indices.tolist()):
                annotations_ijk = task_annotations_list[k]
                mask_ijk = task_mask_list[k].clone()

                num_interior_pts = mask_ijk.sum(dim=-1)
                task_task_full_resolution_interior_pts = (
                    task_full_resolution_interior_pts_list[k]
                )
                if fpn_assignment_method == "POINTS":
                    partitions = targets_config.point_intervals
                    partitions = {1: (0, 64), 2: (65, 512), 4: (512, torch.inf)}
                    lower, upper = partitions[stride]
                    partition_mask = torch.logical_and(
                        task_task_full_resolution_interior_pts > lower,
                        task_task_full_resolution_interior_pts <= upper,
                    )

                    annotations_ijk = annotations_ijk[partition_mask]
                    mask_ijk = mask_ijk[partition_mask]
                    num_interior_pts = num_interior_pts[partition_mask]
                    if num_interior_pts.shape[0] == 0:
                        continue

                _, perm = num_interior_pts.sort(stable=True, descending=False)

                num_interior_pts = num_interior_pts[perm]
                annotations_ijk = annotations_ijk[perm]
                mask_ijk = mask_ijk[perm]

                mask_ijk = mask_ijk.view(-1, strided_height, strided_width)
                instance_ids = (
                    mask_ijk
                    * torch.arange(
                        1,
                        mask_ijk.shape[0] + 1,
                        device=mask_ijk.device,
                        dtype=torch.float32,
                    )[:, None, None]
                )

                # Debug.
                num_seen = instance_ids.unique()[1:].shape[0]
                previous_indices = instance_ids.unique()[1:]
                # print(num_seen)

                instance_ids[mask_ijk.logical_not().nonzero(as_tuple=True)] = torch.inf
                indices, _ = instance_ids.min(dim=0, keepdim=True)
                indices = indices.nan_to_num(posinf=0).long()

                category_ids = annotations_ijk[:, -2].long()
                cats = mask_ijk * category_ids[:, None, None]

                num_categories = len(tasks_config[t_id])
                cats[mask_ijk.logical_not().nonzero(as_tuple=True)] = num_categories

                cats = cats.gather(0, (indices - 1).clamp(0))
                tgts[stride][t_id]["classification_labels"][batch_index] = cats.squeeze(
                    1
                ).long()
                tgts[stride][t_id]["panoptics"][batch_index] = indices

                num_resolved = indices.unique()[1:].shape[0]
                if num_seen != num_resolved:
                    _, counts = mask_ijk.nonzero()[:, 1:].unique(
                        dim=0, return_counts=True
                    )
                    if counts.max() <= 1:
                        print(previous_indices)
                        print(indices.unique()[1:])
                        breakpoint()
                reg_tgts = encode_regression_targets(
                    annotations_ijk,
                    cart_ij,
                    enable_azimuth_invariant_targets,
                )
                reg_tgts = reg_tgts.permute(0, 2, 1).view(
                    -1, reg_tgts.shape[-1], strided_height, strided_width
                )
                reg_tgts = reg_tgts.gather(
                    0,
                    (indices - 1)
                    .clamp(0)[:, None]
                    .repeat_interleave(reg_tgts.shape[1], 1),
                )

                tgts[stride][t_id]["regression_targets"][batch_index] = (
                    reg_tgts * mask_ijk.gather(0, (indices - 1).clamp(0))[:, None]
                )

                points_per_obj = mask_ijk * num_interior_pts[:, None, None]
                points_per_obj = points_per_obj.gather(0, (indices - 1).clamp(0))
                tgts[stride][t_id]["points_per_obj"][batch_index] = points_per_obj
    return tgts


def initialize_targets(
    tasks_cfg: DictConfig,
    fpn_strides: List[int],
    shape: torch.Size,
    device: torch.device,
) -> Dict[int, Dict[int, Dict[str, Tensor]]]:
    num_regressands = 8

    B, _, H, W = shape
    targets: DefaultDict[int, Dict[int, Dict[str, Tensor]]] = defaultdict(dict)
    for stride in fpn_strides:
        stride_h, stride_w = 1, stride
        strided_width = int(W / stride_w)
        strided_height = int(H / stride_h)

        targets[stride] = defaultdict(dict)
        for key, categories in tasks_cfg.items():
            t_id = cast(int, key)
            num_categories = len(categories)
            targets[stride][t_id]["points_per_obj"] = torch.zeros(
                (B, 1, strided_height, strided_width),
                device=device,
                dtype=torch.int64,
            )

            targets[stride][t_id]["panoptics"] = torch.zeros(
                (B, 1, strided_height, strided_width),
                device=device,
                dtype=torch.int64,
            )

            targets[stride][t_id]["classification_labels"] = torch.full(
                size=(B, strided_height, strided_width),
                fill_value=num_categories,
                dtype=torch.long,
                device=device,
            )

            targets[stride][t_id]["regression_targets"] = torch.zeros(
                size=(B, num_regressands, strided_height, strided_width),
                device=device,
            )

            targets[stride][t_id]["num_category"] = torch.ones(
                size=(B, num_categories, 1, 1),
                device=device,
            )
    return targets
