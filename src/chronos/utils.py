# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List
from pandas.tseries.frequencies import to_offset
import torch
from pandas.tseries.offsets import (
    Tick,
    Day, BusinessDay, CDay,
    Week, WeekOfMonth,
    MonthEnd, MonthBegin, BMonthEnd, BMonthBegin,
    QuarterEnd, QuarterBegin, BQuarterEnd, BQuarterBegin,
    YearEnd, YearBegin, BYearEnd, BYearBegin, SemiMonthBegin,
    SemiMonthEnd
)

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
        print(f"Warning: frequency {freq} not recognized. Will return None as frequency id.")
        return None

    if isinstance(offset, (MonthEnd, MonthBegin, BMonthEnd, BMonthBegin,
                             QuarterEnd, QuarterBegin, BQuarterEnd, BQuarterBegin,
                             YearEnd, YearBegin, BYearEnd, BYearBegin, SemiMonthBegin, SemiMonthEnd)):
        return 2
    elif isinstance(offset, (Day, CDay, BusinessDay, Week, WeekOfMonth)):
        return 1
    elif isinstance(offset, Tick):
        return 0 if offset.nanos < to_offset("D").nanos else 1
    else:
        print(f"Warning: frequency {freq} not recognized. Will return None as frequency id.")
        return None

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
        return None
