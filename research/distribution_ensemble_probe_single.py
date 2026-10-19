"""Validation-only probe for robust aggregation of one residual-source family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    # This probe deliberately accesses only cache['val']; it never loads test.
    val = cache["val"]
    sources = [
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in args.logits
    ]
    text = cache["prompts"]["pair"].float()
    val_text = zscore(val["com"].float() @ text.T)
    stacked = torch.stack(sources)
    variants = {
        "mean": stacked.mean(0),
        "median": stacked.median(0).values,
        "trimmed_mean": stacked.sort(0).values[1:-1].mean(0)
        if stacked.shape[0] >= 3
        else stacked.mean(0),
    }
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for name, score in variants.items():
        for delta_name, delta in (("full", score), ("update", score - val_text)):
            for strength in (0.2, 0.5, 0.75, 1.0, 1.25):
                fused = evaluator(
                    val["logits"].float() + strength * delta,
                    val["attr"],
                    val["obj"],
                    val["pair"],
                )
                rows.append(
                    {
                        "variant": name,
                        "delta": delta_name,
                        "strength": strength,
                        "auc_gain": fused["AUC"] - base["AUC"],
                        "hm_gain": fused["best_hm"] - base["best_hm"],
                        "fused_val": fused,
                    }
                )
    rows.sort(key=lambda row: (row["auc_gain"], row["hm_gain"]), reverse=True)
    result = {
        "mode": "exploratory_validation_only",
        "test_read": False,
        "base_val": base,
        "rows": rows,
        "best": rows[0],
        "source_count": len(sources),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"base_val": base, "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
