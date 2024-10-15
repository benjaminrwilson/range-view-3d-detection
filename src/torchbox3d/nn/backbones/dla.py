"""Range View Backbone."""

from dataclasses import dataclass, field
from typing import Dict, cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn

from torchbox3d.nn.blocks import AggregationBlock, BasicBlock, ResidualBlock
from torchbox3d.nn.stems import MetaKernel, RangePartition


@dataclass(unsafe_hash=True)
class RangeBackbone(nn.Module):
    """Range View Net (based on DLA)."""

    in_channels: int
    layers: ListConfig
    out_channels: int

    res1: ResidualBlock = field(init=False)
    res2a: ResidualBlock = field(init=False)
    res2: ResidualBlock = field(init=False)
    res3a: ResidualBlock = field(init=False)
    res3: ResidualBlock = field(init=False)

    agg2: AggregationBlock = field(init=False)
    agg1: AggregationBlock = field(init=False)
    agg2a: AggregationBlock = field(init=False)
    agg3: AggregationBlock = field(init=False)

    def __post_init__(self) -> None:
        """Initialize network modules."""
        super().__init__()
        self.res1 = ResidualBlock(
            self.layers[0], self.layers[0], stride=(1, 1), num_blocks=2
        )
        self.res2a = ResidualBlock(
            self.layers[0],
            self.layers[1],
            stride=(1, 2),
            num_blocks=3,
        )
        self.res2 = ResidualBlock(
            self.layers[1],
            self.layers[2],
            stride=(1, 2),
            num_blocks=3,
        )
        self.res3a = ResidualBlock(
            self.layers[2],
            self.layers[3],
            stride=(1, 2),
            num_blocks=5,
        )
        self.res3 = ResidualBlock(
            self.layers[3],
            self.layers[4],
            stride=(1, 2),
            num_blocks=5,
        )

        # res2 + res3
        # channels[2] + channels[4]
        self.agg2 = AggregationBlock(
            self.layers[2],
            self.layers[4],
            self.layers[2],
            kernel_size=(3, 8),
            stride=(1, 4),
            padding=(1, 2),
            num_blocks=2,
        )
        # res1 + res2
        # channels[0] + channels[2]
        self.agg1 = AggregationBlock(
            self.layers[0],
            self.layers[2],
            self.layers[0],
            kernel_size=(3, 8),
            stride=(1, 4),
            padding=(1, 2),
            num_blocks=2,
        )
        # res2a + agg2
        # channels[1] + channels[2]
        self.agg2a = AggregationBlock(
            self.layers[1],
            self.layers[2],
            self.layers[1],
            kernel_size=(3, 4),
            stride=(1, 2),
            padding=(1, 1),
            num_blocks=1,
        )
        # agg1 + agg2a
        # channels[0] + channels[2]
        self.agg3 = AggregationBlock(
            self.layers[0],
            self.layers[1],
            self.layers[0],
            kernel_size=(3, 4),
            stride=(1, 2),
            padding=(1, 1),
            num_blocks=2,
        )

    def forward(
        self, features: Tensor, cart: Tensor, mask: Tensor
    ) -> Dict[int, Tensor]:
        """Network forward pass."""
        res1 = self.res1(features)
        res2a = self.res2a(res1)
        res2 = self.res2(res2a)
        res3a = self.res3a(res2)
        res3 = self.res3(res3a)

        agg2 = self.agg2(res2, res3)
        agg1 = self.agg1(res1, res2)
        agg2a = self.agg2a(res2a, agg2)
        agg3 = self.agg3(agg1, agg2a)

        agg3 = torch.concatenate([features, agg3], dim=1)
        return {
            1: agg3,
            2: agg2a,
            4: agg2,
            16: res3,
        }


@dataclass(unsafe_hash=True)
class RangeNet(nn.Module):
    """RangeNet implementation."""

    in_channels: int
    layers: ListConfig
    out_channels: int
    projection_kernel_size: int
    dataset_name: str
    num_neighbors: int
    num_layers: int

    stem_type: str

    _net: DictConfig

    stem: nn.Module = field(init=False)
    net: nn.Module = field(init=False)
    compile: bool = False

    def __post_init__(self) -> None:
        """Initialize network modules.

        Args:
            in_channels: Number of input channels.
            layers: List of layer channels.
            out_channels: Number of out channels
            dataset_name: Dataset name for the input data.
        """
        super().__init__()
        if self.stem_type == "META":
            self.stem = MetaKernel(
                in_channels=self.in_channels,
                out_channels=self.layers[0],
                num_neighbors=self.num_neighbors,
                num_layers=self.num_layers,
            )
        elif self.stem_type == "RANGE_PARTITION":
            self.stem = RangePartition(
                in_channels=self.in_channels,
                out_channels=self.layers[0],
                num_neighbors=self.num_neighbors,
                num_layers=self.num_layers,
                projection_kernel_size=self.projection_kernel_size,
            )
        elif self.stem_type == "BASIC":
            self.stem = BasicBlock(
                self.in_channels,
                self.layers[0],
                kernel_size=self.projection_kernel_size,
                project=True,
            )
        else:
            raise NotImplementedError("This stem type is not implemented!")
        self.net = instantiate(self._net)
        if self.compile:
            self.stem = torch.compile(self.stem)
            self.net = torch.compile(self.net)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        """Network forward pass."""
        features = x["features"]
        mask = x["mask"]
        cart = x["cart"]

        if self.stem_type == "META":
            features = cast(Tensor, self.stem(features, cart))
        elif self.stem_type == "RANGE_PARTITION":
            features = cast(Tensor, self.stem(features, cart, mask))
        elif self.stem_type == "BASIC":
            features = cast(Tensor, self.stem(features))
        else:
            raise NotImplementedError("This stem type is not implemented!")
        out = cast(Tensor, self.net.forward(features, cart, mask))
        return out
