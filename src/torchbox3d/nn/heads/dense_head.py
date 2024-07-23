"""Deformable detection heads."""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from pytorch_lightning.core.module import LightningModule
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation


@dataclass(unsafe_hash=True)
class DenseHead(LightningModule):
    """Detection Head."""

    in_channels: int
    out_channels: int
    num_cls: int
    kernel_size: int
    final_kernel_size: int

    num_blocks: int = 4
    prior_prob: Optional[float] = None

    blocks: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        blocks: List[nn.Module] = [
            Conv2dNormActivation(
                self.in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
        ]
        for _ in range(self.num_blocks - 1):
            block = Conv2dNormActivation(
                self.out_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            blocks.append(block)

        blocks.append(
            Conv2dNormActivation(
                self.out_channels,
                self.num_cls,
                kernel_size=self.final_kernel_size,
                norm_layer=None,
                activation_layer=None,
                padding="same",
            )
        )

        self.blocks = nn.Sequential(*blocks)

        # Initialization
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

        if self.prior_prob is not None:
            # Use prior in model initialization to improve stability.
            bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
            torch.nn.init.constant_(self.blocks[-1]._modules["0"].bias, bias_value)

    def forward(self, x: Tensor, cart: Tensor, mask: Tensor) -> Tensor:
        """Network forward pass."""
        return self.blocks(x)


# @dataclass(unsafe_hash=True)
# class DenseHead(LightningModule):
#     """Detection Head."""

#     in_channels: int
#     out_channels: int
#     num_cls: int
#     kernel_size: int
#     final_kernel_size: int

#     num_blocks: int = 4
#     residual: bool = False
#     prior_prob: Optional[float] = None

#     blocks: nn.ModuleList = field(init=False)

#     def __post_init__(self) -> None:
#         """Initialize network modules."""
#         super().__init__()
#         blocks: List[nn.Module] = [
#             Conv2dNormActivation(
#                 self.in_channels,
#                 self.out_channels,
#                 kernel_size=self.kernel_size,
#                 padding="same",
#                 activation_layer=nn.ReLU,
#             )
#         ]
#         for _ in range(self.num_blocks - 1):
#             block = Conv2dNormActivation(
#                 self.out_channels,
#                 self.out_channels,
#                 kernel_size=7,
#                 padding="same",
#                 activation_layer=nn.ReLU,
#                 groups=self.out_channels,
#             )
#             blocks.append(block)

#         blocks.append(
#             Conv2dNormActivation(
#                 self.out_channels,
#                 self.num_cls,
#                 kernel_size=self.final_kernel_size,
#                 norm_layer=None,
#                 activation_layer=None,
#                 padding="same",
#             )
#         )

#         self.blocks = nn.ModuleList(blocks)

#         # Initialization
#         for block in self.blocks:
#             for layer in block:
#                 if isinstance(layer, nn.Conv2d):
#                     torch.nn.init.normal_(layer.weight, std=0.01)
#                     if layer.bias is not None:
#                         torch.nn.init.zeros_(layer.bias)

#         if self.prior_prob is not None:
#             # Use prior in model initialization to improve stability.
#             bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
#             torch.nn.init.constant_(self.blocks[-1]._modules["0"].bias, bias_value)

#     def forward(self, x: Tensor, cart: Tensor, mask: Tensor) -> Tensor:
#         """Network forward pass."""
#         for i, block in enumerate(self.blocks):
#             if i == len(self.blocks) - 1 or not self.residual:
#                 x = block(x)
#             else:
#                 x = (block(x) + x).relu_()
#         return x


@dataclass(unsafe_hash=True)
class DenseGroupHead(LightningModule):
    """Detection Head."""

    in_channels: int
    out_channels: int
    num_cls: int
    kernel_size: int
    final_kernel_size: int

    num_blocks: int = 4
    residual: bool = False
    prior_prob: Optional[float] = None

    # blocks: nn.ModuleList = field(init=False)
    xy: DenseHead = field(init=False)
    z: DenseHead = field(init=False)
    dim: DenseHead = field(init=False)
    rot: DenseHead = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()

        self.xy = DenseHead(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_cls=2,
            kernel_size=self.kernel_size,
            final_kernel_size=self.final_kernel_size,
            num_blocks=self.num_blocks,
            residual=self.residual,
        )

        self.z = DenseHead(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_cls=1,
            kernel_size=self.kernel_size,
            final_kernel_size=self.final_kernel_size,
            num_blocks=self.num_blocks,
            residual=self.residual,
        )

        self.dim = DenseHead(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_cls=3,
            kernel_size=self.kernel_size,
            final_kernel_size=self.final_kernel_size,
            num_blocks=self.num_blocks,
            residual=self.residual,
        )

        self.rot = DenseHead(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_cls=2,
            kernel_size=self.kernel_size,
            final_kernel_size=self.final_kernel_size,
            num_blocks=self.num_blocks,
            residual=self.residual,
        )

    def forward(self, x: Tensor, cart: Tensor, mask: Tensor) -> Tensor:
        """Network forward pass."""
        xy = self.xy(x, cart, mask)
        z = self.z(x, cart, mask)
        dim = self.dim(x, cart, mask)
        rot = self.rot(x, cart, mask)

        regressands = torch.cat([xy, z, dim, rot], dim=1)
        return regressands
