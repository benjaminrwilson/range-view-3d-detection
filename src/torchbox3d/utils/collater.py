"""Collation utilities for dataloaders."""

from collections import defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch

from torchbox3d.structures.data import Data
from torchbox3d.structures.sparse_tensor import SparseTensor


def collate_fn(data_list: Sequence[Data]) -> Data:
    """Collate (merge) a sequence of items.

    Args:
        data_list: Sequence of data to be collated.

    Returns:
        The collated data.

    Raises:
        TypeError: If the data type is not supported for collation.
    """
    collated_data: DefaultDict[str, List[Any]] = defaultdict(list)
    for data in data_list:
        for attr_name, attr_list in data.items():
            collated_data[attr_name].append(attr_list)

    output: Dict[str, Any] = {}
    for attr_name, attr_list in collated_data.items():
        elem = attr_list[0]
        if isinstance(elem, pl.DataFrame):
            # Add batch index.
            attr_list = [
                e.with_columns(batch_index=pl.lit([i] * len(e)))
                for (i, e) in enumerate(attr_list)
            ]
            output[attr_name] = pl.concat(attr_list)
        elif isinstance(elem, pd.DataFrame):
            # Add batch index.
            batch_index_list = list(
                chain.from_iterable([[i] * len(e) for i, e in enumerate(attr_list)])
            )
            batched_frame = pd.concat(attr_list).reset_index(drop=True)
            batched_frame["batch_index"] = np.array(batch_index_list)
            output[attr_name] = batched_frame
        elif isinstance(elem, torch.Tensor):
            output[attr_name] = torch.stack(attr_list).pin_memory()
    return output


def sparse_collate(sparse_tensor_list: List[SparseTensor]) -> SparseTensor:
    """Collate a list of sparse tensors.

    Args:
        sparse_tensor_list: List of sparse tensors.

    Returns:
        Collated sparse tensor.
    """
    indices_list = []
    values_list = []
    counts_list = []
    stride = sparse_tensor_list[0].stride

    for key, sparse_tensor in enumerate(sparse_tensor_list):
        input_size = sparse_tensor.indices.shape[0]
        batch = torch.full(
            (input_size, 1),
            key,
            device=sparse_tensor.indices.device,
            dtype=torch.int,
        )

        indices_list.append(torch.cat((sparse_tensor.indices, batch), dim=1))
        values_list.append(sparse_tensor.values)
        counts_list.append(sparse_tensor.counts)

    indices = torch.cat(indices_list, dim=0)
    values = torch.cat(values_list, dim=0)
    counts = torch.cat(counts_list, dim=0)
    output = SparseTensor(indices=indices, values=values, counts=counts, stride=stride)
    return output
