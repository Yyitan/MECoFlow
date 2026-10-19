"""Validation-only aggregation for a frozen diagonal-source ensemble."""

from __future__ import annotations

import argparse
import hashlib
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=1.25)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Deliberately only touch cache['val'] and logits['val'].
    val = cache["val"]
    saved = [torch.load(p, map_location="cpu", weights_only=True) for p in args.classifier_logits]
    if any(set(item) != {"val"} for item in saved):
        raise ValueError("validation-only aggregation requires exact logits keys {'val'}")
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    scores = [zscore(item["val"].float()) for item in saved]
    score_mean = torch.stack(scores).mean(0)
    delta = score_mean - val_text
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused = evaluator(val["logits"].float() + args.strength * delta, val["attr"], val["obj"], val["pair"])
    individual = []
    for index, score in enumerate(scores):
        stats = evaluator(val["logits"].float() + args.strength * (score - val_text), val["attr"], val["obj"], val["pair"])
        individual.append({
            "seed_index": index,
            "fused_val": stats,
            "auc_gain": stats["AUC"] - base["AUC"],
            "hm_gain": stats["best_hm"] - base["best_hm"],
        })
    result = {
        "protocol": {
            "mode": "validation_only_fixed_candidate",
            "candidate": "diagonal_source_mean_update",
            "strength": args.strength,
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "classifier_logits": [{"path": str(Path(p).resolve()), "sha256": sha256(Path(p).resolve())} for p in args.classifier_logits],
            "test_read": False,
        },
        "base_val": base,
        "fused_val": fused,
        "individual": individual,
        "mean": {
            "ensemble_auc_gain": fused["AUC"] - base["AUC"],
            "ensemble_hm_gain": fused["best_hm"] - base["best_hm"],
            "individual_auc_gain_mean": float(np.mean([r["auc_gain"] for r in individual])),
            "individual_auc_gain_std": float(np.std([r["auc_gain"] for r in individual])),
            "individual_hm_gain_mean": float(np.mean([r["hm_gain"] for r in individual])),
            "individual_hm_gain_std": float(np.std([r["hm_gain"] for r in individual])),
            "positive_individual_auc_runs": int(sum(r["auc_gain"] > 0 for r in individual)),
            "positive_individual_hm_runs": int(sum(r["hm_gain"] > 0 for r in individual)),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
