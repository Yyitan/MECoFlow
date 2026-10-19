"""Aggregate validation-gated warm-flow runs without using test data for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in args.runs:
        dataset, path = item.split("=", 1)
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        gain = result["selection"]["val_auc"] - result["base_val"]["AUC"]
        enabled = gain >= args.min_val_auc_gain
        chosen = result["fused_test"] if enabled else result["base_test"]
        grouped.setdefault(dataset, []).append(
            {
                "path": path,
                "enabled": enabled,
                "val_auc_gain": gain,
                "base_test": result["base_test"],
                "selected_test": chosen,
            }
        )
    output: dict[str, object] = {"min_val_auc_gain": args.min_val_auc_gain, "datasets": {}}
    for dataset, runs in grouped.items():
        base_auc = np.array([x["base_test"]["AUC"] for x in runs])
        base_hm = np.array([x["base_test"]["best_hm"] for x in runs])
        auc = np.array([x["selected_test"]["AUC"] for x in runs])
        hm = np.array([x["selected_test"]["best_hm"] for x in runs])
        output["datasets"][dataset] = {
            "num_runs": len(runs),
            "enabled_runs": int(sum(x["enabled"] for x in runs)),
            "base_auc_mean": float(base_auc.mean()),
            "base_hm_mean": float(base_hm.mean()),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
            "hm_mean": float(hm.mean()),
            "hm_std": float(hm.std(ddof=1)) if len(hm) > 1 else 0.0,
            "auc_gain_points": float(100.0 * (auc.mean() - base_auc.mean())),
            "hm_gain_points": float(100.0 * (hm.mean() - base_hm.mean())),
            "runs": runs,
        }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
