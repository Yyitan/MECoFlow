"""Validation-only probe for robustly combining existing residual sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--iso-logits", nargs="+", required=True)
    parser.add_argument("--diag-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    # Deliberately access only cache['val']; this probe never touches test.
    val = cache["val"]
    iso = [zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in args.iso_logits]
    diag = [zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in args.diag_logits]
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    variants = {
        "iso_mean": torch.stack(iso).mean(0),
        "diag_mean": torch.stack(diag).mean(0),
        "source_mean": torch.stack(iso + diag).mean(0),
        "source_median": torch.stack(iso + diag).median(0).values,
        "source_trimmed_mean": torch.stack(iso + diag).sort(0).values[1:-1].mean(0),
    }
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for name, score in variants.items():
        for delta_name, delta in (("full", score), ("update", score - val_text)):
            for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
                fused = evaluator(val["logits"].float() + strength * delta, val["attr"], val["obj"], val["pair"])
                rows.append({
                    "variant": name,
                    "delta": delta_name,
                    "strength": strength,
                    "auc_gain": fused["AUC"] - base["AUC"],
                    "hm_gain": fused["best_hm"] - base["best_hm"],
                    "fused_val": fused,
                })
    rows.sort(key=lambda r: (r["auc_gain"], r["hm_gain"]), reverse=True)
    result = {
        "mode": "exploratory_validation_only",
        "test_read": False,
        "base_val": base,
        "rows": rows,
        "best": rows[0],
        "source_count": {"isotropic": len(iso), "diagonal": len(diag)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"base_val": base, "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
