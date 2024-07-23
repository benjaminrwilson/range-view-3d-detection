"""Methods to help visualize data during training."""

import math
from typing import Any, Dict, Final, Optional, Tuple, cast

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from kornia.enhance.normalize import normalize_min_max
from kornia.geometry.conversions import euler_from_quaternion
from matplotlib.cm import get_cmap
from mmcv.ops import boxes_iou3d
from omegaconf import DictConfig
from polars import col
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torchvision.io.image import write_png

from torchbox3d.math.polytope import cuboids_to_vertices

GREYS: Final = (
    torch.stack([torch.as_tensor(get_cmap("binary_r")(i)[:3]) for i in range(256)])
    .mul(255.0)
    .byte()
    .t()
)
TURBO_R: Final = (
    torch.stack([torch.as_tensor(get_cmap("turbo_r")(i)[:3]) for i in range(256)])
    .mul(255.0)
    .byte()
    .t()
)
VIRIDIS: Final = (
    torch.stack([torch.as_tensor(get_cmap("viridis")(i)[:3]) for i in range(256)])
    .mul(255.0)
    .byte()
    .t()
)

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
)


@rank_zero_only
def to_logger(
    dts: pl.DataFrame,
    uuids: pl.DataFrame,
    data: Dict[str, Any],
    network_outputs: Dict[int, Dict[str, Tensor]],
    tasks_cfg: DictConfig,
    trainer: Trainer,
    debug: bool,
    batch_index: int,
    max_num_dts: int = 100,
    max_side: float = 75.0,
    meter_per_px: float = 0.15,
) -> None:
    """Log training, validation, etc. information to TensorBoard.

    Args:
        dts: Detections.
        data: Ground truth targets.
        network_outputs: Encoded outputs from the network.
        trainer: PytorchLightning Trainer class.
        prepreocessing_config: Preprocessing configuration.
        max_num_dts: Max number of detections to display.
        max_side: Max side of the bird's-eye view image.
        meter_per_px: Meters per pixel in the bird's-eye view.
    """
    if trainer.state.stage == RunningStage.SANITY_CHECKING:
        return
    img = draw_detections(
        dts=dts,
        uuids=uuids,
        data=data,
        network_outputs=network_outputs,
        max_num_dts=max_num_dts,
        max_side=max_side,
        meter_per_px=meter_per_px,
    )
    log_id, timestamp_ns, _ = uuids.filter(pl.col("batch_index") == batch_index).row(0)
    prefix = f"{log_id}-{timestamp_ns}"
    log_image(name=prefix, img=img, trainer=trainer)


def log_image(name: str, img: np.array, trainer: Trainer) -> None:
    """Log an image.

    Args:
        name: Displayed image name.
        img: (C,H,W) Image tensor.
        trainer: Pytorch-lightning trainer class.
    """
    if trainer.logger is None:
        return

    stage = trainer.state.stage
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.log_image(f"{stage}", [img], caption=[name])
    elif isinstance(trainer.logger, TensorBoardLogger):
        img = img.transpose(2, 0, 1)
        stage = trainer.state.stage
        trainer.logger.experiment.add_image(
            f"{stage}/{name}",
            img,
            global_step=trainer.global_step,
        )


def draw_on_bev(
    dts: pl.DataFrame,
    img: np.array,
    color: Tuple[int, int, int] = (0, 255, 0),
    max_side: float = 50.0,
    meter_per_px: float = 0.1,
    colors: Optional[np.array] = None,
) -> np.ndarray:
    """Draw a set of bounding boxes on a BEV image.

    Args:
        grid: Object describing voxel grid characteristics.
        img: (3,H,W) Bird's-eye view image.
        color: 3-channel color (RGB or BGR).

    Returns:
        (3,H,W) Image with boxes drawn.
    """
    cuboids = (
        dts.select(
            [
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
            ],
        )
        .to_numpy()
        .astype(np.float32)
    )

    scores = np.ones_like(cuboids[:, 0])

    if colors is None:
        colors = np.full((len(cuboids), 3), fill_value=(0, 0, 255))
    if "score" in dts.columns:
        scores = dts["score"].to_numpy()

    cuboids = torch.as_tensor(cuboids)
    yaw = euler_from_quaternion(*cuboids[:, 6:10].t().split(1))[-1].t()
    cuboids = torch.concatenate([cuboids[:, :6], yaw], dim=1)
    vertices = cuboids_to_vertices(cuboids[None])[0, :, [6, 2, 3, 7], :2]
    for i, (vertices, score) in enumerate(zip(vertices, scores)):
        vertices = world_to_image(vertices, max_side, meter_per_px).numpy()[:, [1, 0]]
        img = cv2.polylines(
            img=img,
            pts=vertices[None].astype(np.int32),
            isClosed=False,
            color=(colors[i] * score).tolist(),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return img


def build_bev(xyz: torch.Tensor, max_side: float, meter_per_px: float) -> torch.Tensor:
    mask = xyz.norm(dim=-1) > 0
    xyz = xyz[mask]
    size = (
        int(math.ceil(max_side * 2 / meter_per_px)),
        int(math.ceil(max_side * 2 / meter_per_px)),
        3,
    )
    x, y = world_to_image(xyz, max_side, meter_per_px)[:, :2].t().long()

    bev = torch.zeros(size, dtype=torch.uint8)
    bev[x, y, :3] = 255
    return bev


def world_to_image(xyz: torch.Tensor, max_side: int, meter_per_px: int) -> torch.Tensor:
    xyz[:, :2] = (xyz[:, :2] + max_side) / meter_per_px
    mask = torch.logical_and(
        xyz[:, :2] > 0, xyz[:, :2] < max_side / meter_per_px * 2.0
    ).all(dim=-1)
    xyz = xyz[mask]

    size = max_side * 2.0 / meter_per_px
    xyz[:, :2] = torch.floor(size - xyz[:, :2]).long()
    return xyz


def normalize_and_color(
    img: np.ndarray, mask: np.ndarray, cmap: np.ndarray
) -> np.ndarray:
    normalized_img = np.clip(255.0 * img / img.max(), 0.0, 255.0).astype(np.uint8)
    return cmap[normalized_img][:, :, :3] * mask


def draw_text(img, label, text_color, x1, y1, size, thickness):
    label = str(label)

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)

    # Prints the text.
    img = cv2.rectangle(img, (x1, y1 + h), (x1 + w, y1), (0.0, 0.0, 0.0), -1)
    img = cv2.putText(
        img,
        label,
        (x1, y1 + h),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return img


def write_debug(img: Tensor, dst: str, mask: Optional[Tensor] = None) -> None:
    img = img.detach().clone().float()
    img = torch.cat(img.split(1, 0), dim=2)
    img = normalize_apply_colormap(img, VIRIDIS.to(img.device), mask=mask).cpu()
    write_png(
        img,
        dst,
    )


def draw_detections(
    dts: pl.DataFrame,
    uuids: pl.DataFrame,
    data: Dict[str, Any],
    network_outputs: Dict[int, Dict[str, Tensor]],
    max_num_dts: int = 500,
    max_side: float = 75.0,
    meter_per_px: float = 0.15,
) -> Tensor:
    batch_index = 0

    image_list = []
    for batch_index in [batch_index]:
        annotations = data["annotations"].filter(pl.col("batch_index") == batch_index)
        sel_dts = dts.filter(col("batch_index") == batch_index).top_k(
            max_num_dts, by="score"
        )

        multiscale_imgs_list = []
        for key, multiscale_outputs in network_outputs["head"].items():
            stride = cast(int, key)

            cart = multiscale_outputs["cart"][batch_index].detach()
            mask = multiscale_outputs["mask"][batch_index].detach()

            dists = torch.linalg.norm(cart, dim=0, keepdim=True)
            dists = normalize_apply_colormap(
                img=dists.unsqueeze(0),
                colormap=TURBO_R.to(dists.device),
                mask=mask,
            )

            if stride == 1:
                xyz = torch.as_tensor(cart.reshape(3, -1).transpose(1, 0).clone())
                bev = build_bev(xyz, max_side, meter_per_px).numpy()
                if annotations.shape[0] > 0:
                    bev = draw_on_bev(
                        annotations,
                        bev,
                        max_side=max_side,
                        meter_per_px=meter_per_px,
                    ).astype(np.uint8)
                if sel_dts.shape[0] > 0 and annotations.shape[0] > 0:
                    dts_tch = torch.as_tensor(
                        sel_dts.select(
                            COLS,
                        ).to_numpy(writable=True)
                    ).cuda()

                    gts_tch = torch.as_tensor(
                        annotations.select(
                            COLS,
                        ).to_numpy(writable=True)
                    ).cuda()

                    _, _, yaws_pds = euler_from_quaternion(*dts_tch[:, 6:10].t())
                    _, _, yaws_gts = euler_from_quaternion(*gts_tch[:, 6:10].t())

                    dts_tch = torch.cat([dts_tch[:, :6], yaws_pds[:, None]], dim=-1)
                    gts_tch = torch.cat([gts_tch[:, :6], yaws_gts[:, None]], dim=-1)

                    ious = boxes_iou3d(dts_tch.float(), gts_tch.float())
                    max_ious, _ = ious.max(dim=-1)
                    max_ious = max_ious.cpu().numpy()
                    colors = np.full((len(max_ious), 3), fill_value=(255, 0, 0))
                    colors[max_ious >= 0.7] = np.array([0, 255, 0])
                    bev = draw_on_bev(
                        sel_dts,
                        bev,
                        max_side=max_side,
                        meter_per_px=meter_per_px,
                        colors=colors,
                    ).astype(np.uint8)

                bev = torch.as_tensor(bev).permute(2, 0, 1)
                outputs = network_outputs["head"][stride]
                for task_id, task_outputs in outputs.items():
                    if not isinstance(task_id, int):
                        continue
                    likelihoods, _ = (
                        task_outputs["logits"][batch_index]
                        .sigmoid()
                        .max(dim=0, keepdim=True)
                    )

                    likelihoods = normalize_apply_colormap(
                        img=likelihoods.unsqueeze(0),
                        colormap=VIRIDIS.to(likelihoods.device),
                        mask=mask,
                    )

                    C, H, W = dists.shape
                    _, SH, SW = bev.shape

                    left = int(math.fabs(W - SW) // 2)
                    right = int(math.fabs(W - SW) - left)
                    if W < SW:
                        likelihoods = F.pad(likelihoods, [left, right, 0, 0])

                    multiscale_imgs_list.append(likelihoods.cpu())

                if "aux" in network_outputs:
                    outputs = network_outputs["aux"][stride]
                    for task_id, task_outputs in outputs.items():
                        if not isinstance(task_id, int):
                            continue
                        for k, v in task_outputs.items():
                            for i, loss in enumerate(v.abs()[batch_index]):
                                H, W = loss.shape
                                loss = normalize_apply_colormap(
                                    img=loss.view(1, 1, H, W).float(),
                                    colormap=VIRIDIS.to(likelihoods.device),
                                    mask=mask,
                                )

                                loss = draw_text(
                                    np.ascontiguousarray(
                                        loss.permute(1, 2, 0).cpu().numpy()
                                    ),
                                    f"{k}_{i}",
                                    (0, 255, 0),
                                    0,
                                    0,
                                    0.5,
                                    2,
                                )
                                loss = torch.as_tensor(loss).permute(2, 0, 1)

                                _, _, SW = bev.shape
                                left = int(math.fabs(W - SW) // 2)
                                right = int(math.fabs(W - SW) - left)
                                if W < SW:
                                    loss = F.pad(loss, [left, right, 0, 0])

                                multiscale_imgs_list.append(loss.cpu())

            C, H, W = dists.shape
            _, SH, SW = bev.shape

            left = int(math.fabs(W - SW) // 2)
            right = int(math.fabs(W - SW) - left)

            if W < SW:
                dists = F.pad(dists, [left, right, 0, 0])
            else:
                bev = F.pad(bev, [left, right, 0, 0])
            multiscale_imgs_list.append(dists.cpu())

            outputs = network_outputs["head"][stride]
            for task_id, task_outputs in outputs.items():
                if not isinstance(task_id, int):
                    continue
                likelihoods, _ = (
                    task_outputs["logits"][batch_index]
                    .sigmoid()
                    .max(dim=0, keepdim=True)
                )

                likelihoods = normalize_apply_colormap(
                    img=likelihoods.unsqueeze(0),
                    colormap=VIRIDIS.to(likelihoods.device),
                    mask=mask,
                )

                C, H, W = likelihoods.shape
                _, SH, SW = bev.shape

                left = int(math.fabs(W - SW) // 2)
                right = int(math.fabs(W - SW) - left)
                if W < SW:
                    likelihoods = F.pad(likelihoods, [left, right, 0, 0])

                multiscale_imgs_list.append(likelihoods.cpu())

            # for (
            #     task_id,
            #     _,
            # ) in network_outputs["head"].items():

            #     breakpoint()
            #     classification_scores = (
            #         network_outputs["head"][stride][task_id]["logits"]
            #         .detach()
            #         .sigmoid()
            #     )

        img = torch.cat(multiscale_imgs_list + [bev], dim=1)
        image_list.append(img)
    return torch.cat(image_list)


def normalize_apply_colormap(
    img: Tensor, colormap: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    assert img.ndim == 4

    _, _, H, W = img.shape
    img = normalize_min_max(img).mul(255.0).long().view(H, W)
    img = colormap[:, img]
    if mask is not None:
        img *= mask
    return img
