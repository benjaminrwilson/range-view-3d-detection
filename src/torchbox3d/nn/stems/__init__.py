from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation

from torchbox3d.nn.blocks import BasicBlock


@dataclass(unsafe_hash=True)
class MetaKernel(nn.Module):
    """MetaKernel implementation."""

    in_channels: int
    out_channels: int
    num_neighbors: int
    num_layers: int = 2

    projection: BasicBlock = field(init=False)
    positional_kernel: nn.Sequential = field(init=False)
    fusion_kernel: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules.

        Args:
            in_channels: Number of input channels.
            layers: List of layer channels.
            out_channels: Number of out channels
            dataset_name: Dataset name for the input data.
        """
        super().__init__()
        self.projection = BasicBlock(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            project=True,
        )
        self.positional_kernel = nn.Sequential(
            *[
                Conv2dNormActivation(
                    3 if i == 0 else self.out_channels, self.out_channels, kernel_size=1
                )
                for i in range(self.num_layers)
            ]
        )
        self.fusion_kernel = nn.Sequential(
            *[
                Conv2dNormActivation(
                    (
                        self.out_channels * self.num_neighbors**2
                        if i == 0
                        else self.out_channels
                    ),
                    self.out_channels,
                    kernel_size=1,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, features: Tensor, cart: Tensor) -> Tensor:
        """Compute the meta kernel features."""
        features = self.projection(features)

        B, _, H, W = features.shape
        padding = self.num_neighbors // 2
        feature_encoding = F.unfold(features, self.num_neighbors, padding=padding).view(
            B, -1, self.num_neighbors**2, H * W
        )

        cartesian_coordinates = F.unfold(
            cart, self.num_neighbors, padding=padding
        ).view(B, -1, self.num_neighbors**2, H * W)

        center_idx = int(self.num_neighbors**2 / 2)
        cartesian_anchor = cartesian_coordinates[:, :, center_idx : center_idx + 1]
        relative_coordinates = cartesian_coordinates - cartesian_anchor

        positional_encoding: Tensor = self.positional_kernel(relative_coordinates)
        geometric_features = (positional_encoding * feature_encoding).view(B, -1, H, W)
        geometric_features: Tensor = self.fusion_kernel(geometric_features)
        return geometric_features


@dataclass(unsafe_hash=True)
class RangePartition(nn.Module):
    in_channels: int
    out_channels: int
    num_neighbors: int
    projection_kernel_size: int

    num_layers: int = 2

    projection: BasicBlock = field(init=False)
    lower_bounds: nn.Parameter = field(init=False)
    upper_bounds: nn.Parameter = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.projection = BasicBlock(
            6 * self.in_channels,
            self.out_channels,
            kernel_size=self.projection_kernel_size,
            project=True,
        )

        self.lower_bounds = nn.Parameter(
            torch.as_tensor((0, 10, 15, 20, 30, 45)).view(1, -1, 1, 1),
            requires_grad=False,
        )
        self.upper_bounds = nn.Parameter(
            torch.as_tensor((15, 20, 30, 40, 60, torch.inf)).view(1, -1, 1, 1),
            requires_grad=False,
        )

    def forward(self, features: Tensor, cart: Tensor, mask: Tensor) -> Tensor:
        dists = cart.norm(dim=1, keepdim=True)
        partitions = torch.logical_and(
            dists >= self.lower_bounds, dists <= self.upper_bounds
        )

        features = partitions[:, :, None] * features[:, None]

        features = features.flatten(1, 2) * mask
        return cast(Tensor, self.projection(features))
