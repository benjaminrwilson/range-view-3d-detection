"""Subpackage for network blocks."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from torchbox3d.nn.modules.conv import Conv2dSame


@dataclass(unsafe_hash=True)
class BasicBlock(nn.Module):
    """Basic block for feature extraction."""

    in_channels: int
    out_channels: int
    stride: _size_2_t = 1
    dilation: _size_2_t = 1
    kernel_size: _size_2_t = 3
    project: bool = False

    net: nn.Sequential = field(init=False)
    projection_block: Optional[nn.Sequential] = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        """Initialize network modules."""
        super().__init__()
        self.net = nn.Sequential(
            Conv2dSame(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                bias=False,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            Conv2dSame(
                self.out_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=False,
                dilation=self.dilation,
            ),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.project:
            self.projection_block = nn.Sequential(
                Conv2dSame(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                    dilation=self.dilation,
                ),
                nn.BatchNorm2d(self.out_channels),
            )
        else:
            self.projection_block = None

    def forward(self, x: Tensor, residual: Optional[Tensor] = None) -> Tensor:
        """Network forward pass.

        Args:
            x: BCHW Tensor of network inputs.
            residual: BCHW Tensor of residual network inputs.

        Returns:
            Network features.
        """
        residual = x if residual is None else residual
        if self.projection_block is not None:
            residual = self.projection_block(residual)
        return F.relu_(self.net(x) + residual)


@dataclass(unsafe_hash=True)
class ResidualBlock(nn.Module):
    """Extraction block."""

    in_channels: int
    out_channels: int
    num_blocks: int
    stride: _size_2_t = 1
    dilation: _size_2_t = 1
    kernel_size: _size_2_t = 3

    blocks: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        blocks: List[nn.Module] = [
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dilation=self.dilation,
                kernel_size=self.kernel_size,
                stride=self.stride,
                project=True,
            )
        ]
        for _ in range(2, self.num_blocks + 1):
            block = BasicBlock(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dilation=self.dilation,
                kernel_size=self.kernel_size,
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Network forward pass."""
        x = self.blocks(x)
        return x


@dataclass(unsafe_hash=True)
class AggregationBlock(nn.Module):
    """Aggregation block."""

    in_channels_x1: int
    in_channels_x2: int
    out_channels: int
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    num_blocks: int

    upscale: nn.Module = field(init=False)
    normalization: nn.Module = field(init=False)
    activation: nn.Module = field(init=False)
    block: ResidualBlock = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            in_channels=self.in_channels_x2,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )

        self.normalization = nn.BatchNorm2d(num_features=self.out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.block = ResidualBlock(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_blocks=self.num_blocks,
        )

    def forward(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        """Network forward pass.

        Args:
            x_1: BCHW tensor.
            x_2: BCHW tensor.

        Returns:
            Network features.
        """
        x_2 = self.upscale(x_2)
        x_2 = self.normalization(x_2)
        x_2 = self.activation(x_2)

        x_1 = x_1 + x_2
        x_1 = self.block(x_1)
        return x_1
