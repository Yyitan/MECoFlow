#!/usr/bin/env python3
"""Aggregate fixed-shared-selection AOAR test runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(values: list[float]) -> dict[str, object]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "per_seed": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if len(runs) < 2:
        parser.error("at least two test runs are required")

    selections = [run["selection"] for run in runs]
    fixed = {(item["delta"], float(item["strength"]), item["source"]) for item in selections}
    if len(fixed) != 1:
        parser.error("runs do not share one fixed validation selection")
    if any("fused_test" not in run or "base_test" not in run for run in runs):
        parser.error("input lacks test metrics")

    base_auc = [float(run["base_test"]["AUC"]) for run in runs]
    base_hm = [float(run["base_test"]["best_hm"]) for run in runs]
    fused_auc = [float(run["fused_test"]["AUC"]) for run in runs]
    fused_hm = [float(run["fused_test"]["best_hm"]) for run in runs]
    delta, strength, source = next(iter(fixed))
    output = {
        "protocol": "fixed shared validation selection evaluated on test",
        "inputs": [str(path) for path in args.inputs],
        "selection": {"delta": delta, "strength": strength, "source": source},
        "base_test": {"AUC": summarize(base_auc), "best_hm": summarize(base_hm)},
        "fused_test": {"AUC": summarize(fused_auc), "best_hm": summarize(fused_hm)},
        "delta_test": {
            "AUC": {
                **summarize([value - base for value, base in zip(fused_auc, base_auc)]),
                "all_positive": all(value > base for value, base in zip(fused_auc, base_auc)),
            },
            "best_hm": {
                **summarize([value - base for value, base in zip(fused_hm, base_hm)]),
                "all_positive": all(value > base for value, base in zip(fused_hm, base_hm)),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["delta_test"], indent=2))


if __name__ == "__main__":
    main()
