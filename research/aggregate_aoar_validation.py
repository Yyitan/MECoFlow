#!/usr/bin/env python3
"""Aggregate validation-only AOAR runs without reading test predictions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if len(runs) < 2:
        parser.error("at least two validation runs are required")
    if any("fused_test" in run or "base_test" in run for run in runs):
        parser.error("input contains test metrics; expected validation-only results")

    trial_keys = list(runs[0]["trials"])
    expected = set(trial_keys)
    if any(set(run["trials"]) != expected for run in runs[1:]):
        parser.error("trial grids differ between seeds")

    aggregate_trials = {}
    best_key = None
    best_tuple = None
    for key in trial_keys:
        aucs = [float(run["trials"][key]["AUC"]) for run in runs]
        hms = [float(run["trials"][key]["best_hm"]) for run in runs]
        aggregate_trials[key] = {
            "auc_mean": mean(aucs),
            "auc_sample_std": sample_std(aucs),
            "hm_mean": mean(hms),
            "hm_sample_std": sample_std(hms),
            "per_seed_auc": aucs,
            "per_seed_hm": hms,
        }
        candidate = (mean(aucs), mean(hms), key)
        if best_tuple is None or candidate > best_tuple:
            best_tuple = candidate
            best_key = key

    assert best_key is not None and best_tuple is not None
    base_aucs = [float(run["base_val"]["AUC"]) for run in runs]
    base_hms = [float(run["base_val"]["best_hm"]) for run in runs]
    selected = aggregate_trials[best_key]
    gain = selected["auc_mean"] - mean(base_aucs)
    enabled = gain >= args.min_val_auc_gain
    delta, strength_text = best_key.rsplit("_", 1)
    output = {
        "protocol": "shared validation-only selection across generator seeds",
        "inputs": [str(path) for path in args.inputs],
        "num_seeds": len(runs),
        "base_val": {
            "auc_mean": mean(base_aucs),
            "auc_sample_std": sample_std(base_aucs),
            "hm_mean": mean(base_hms),
            "hm_sample_std": sample_std(base_hms),
        },
        "shared_selection": {
            "trial": best_key,
            "delta": delta,
            "strength": float(strength_text),
            **selected,
            "val_auc_gain": gain,
            "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": enabled,
        },
        "independent_seed_best_diagnostic": [run["selection"] for run in runs],
        "trials": aggregate_trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["shared_selection"], indent=2))


if __name__ == "__main__":
    main()
