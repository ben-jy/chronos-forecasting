# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List
from pandas.tseries.frequencies import to_offset
import torch


def left_pad_and_stack_1D(tensors: List[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)

def get_frequency_id(freq: str) -> int:
    try:
        offset = to_offset(freq)
    except ValueError:
        raise ValueError(f"Invalid frequency string: {freq}")

    rule = offset.rule_code

    high_freq = {"ns", "us", "ms", "s", "min", "h"}
    mid_freq = {"D", "B", "W", "W-MON", "W-SUN"}

    if rule in high_freq:
        return 0
    elif rule in mid_freq:
        return 1
    else:
        return 2

def get_domain_id(domain: str) -> int:
    if domain == "transport":
        return 0
    elif domain == "weather":
        return 1
    elif domain == "energy":
        return 2
    elif domain == "web":
        return 3
    else:
        # basically means any
        return None
