"""Residual networks."""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    ModuleList,
    ReLU,
    Sequential,
)


@dataclass(unsafe_hash=True)
class ResNet(LightningModule):
    """Construct a residual network object.

    Args:
        name: Name of the module.
        in_channels: Number of input channels.
        down_strides: Stride for each block.
        down_planes: Number of filters for each block.
        layer_nums: Number of layers per block.
        up_strides: Stride during upsampling blocks.
        num_up_filters: Number of upsampling filters per upsampling block.
    """

    name: str
    in_channels: int
    down_strides: List[int]
    down_planes: List[int]
    layer_nums: List[int]
    up_strides: List[int]
    num_up_filters: List[int]

    up_start_idx: int = field(init=False)
    down_blocks: ModuleList = field(init=False)
    up_blocks: ModuleList = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.up_start_idx = len(self.layer_nums) - len(self.up_strides)

        in_filters = [self.in_channels, *list(self.down_planes)[:-1]]

        down_blocks: List[Sequential] = []
        up_blocks: List[Sequential] = []
        for i, idx in enumerate(self.layer_nums):
            down_block, num_out_filters = _build(
                in_filters[i],
                self.down_planes[i],
                idx,
                stride=self.down_strides[i],
            )
            down_blocks.append(down_block)
            if i - self.up_start_idx >= 0:
                stride = self.up_strides[i - self.up_start_idx]
                if stride > 1:
                    up_block = Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            self.num_up_filters[i - self.up_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        BatchNorm2d(
                            self.num_up_filters[i - self.up_start_idx]
                        ),  # type: ignore
                        ReLU(inplace=True),
                    )
                else:
                    stride = round(1 / stride)
                    up_block = Sequential(
                        Conv2d(
                            num_out_filters,
                            self.num_up_filters[i - self.up_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        BatchNorm2d(
                            self.num_up_filters[i - self.up_start_idx]
                        ),  # type: ignore
                        ReLU(inplace=True),
                    )
                up_blocks.append(up_block)
        self.down_blocks = ModuleList(down_blocks)
        self.up_blocks = ModuleList(up_blocks)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Network forward pass.

        Args:
            x: (B,C,H,W) Tensor of network inputs.

        Returns:
            (B,C,H,W) Tensor of features.
        """
        feats: List[Tensor] = []
        for i, _ in enumerate(self.down_blocks):
            x = self.down_blocks[i](x)
            if i - self.up_start_idx >= 0:
                feats.append(self.up_blocks[i - self.up_start_idx](x))
        if len(feats) > 0:
            x = torch.cat(feats, dim=1)
        return x


def _build(
    in_planes: int, out_planes: int, num_blocks: int, stride: int = 1
) -> Tuple[Sequential, int]:
    """Build the residual network.

    Args:
        in_planes: The number of input channels.
        out_planes: The number of output channels.
        num_blocks: The number of blocks.
        stride: The stride of the block.

    Returns:
        Blocks and the number of output channels.
    """
    blocks = ModuleList([])
    blocks.append(
        Sequential(
            Conv2d(in_planes, out_planes, 3, stride=stride, bias=False, padding=1),
            BatchNorm2d(out_planes),  # type: ignore
            ReLU(inplace=True),
        )
    )

    for _ in range(num_blocks):
        blocks.append(Conv2d(out_planes, out_planes, 3, padding=1, bias=False))
        blocks.append(BatchNorm2d(out_planes))  # type: ignore
        blocks.append(ReLU(inplace=True))

    return Sequential(*blocks), out_planes
