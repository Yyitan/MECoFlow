"""Source-seed audit wrapper for sigmoid/hard entropy schedules."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import audit_adaptive_route_source_seeds as base


def schedule_alpha(candidate: dict[str, object], uncertainty: torch.Tensor) -> torch.Tensor:
    name = str(candidate["schedule"])
    match = re.fullmatch(r"unseen_entropy_([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return uncertainty.pow(float(match.group(1)))
    match = re.fullmatch(r"sigmoid_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return torch.sigmoid(float(match.group(1)) * (uncertainty - float(match.group(2))))
    match = re.fullmatch(r"hard_([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return (uncertainty >= float(match.group(1))).float()
    match = re.fullmatch(r"reverse_hard_([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return (uncertainty <= float(match.group(1))).float()
    raise ValueError(f"unsupported schedule: {name}")


base.schedule_alpha = schedule_alpha
base.main()
