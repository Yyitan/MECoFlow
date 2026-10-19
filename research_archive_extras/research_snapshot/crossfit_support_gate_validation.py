"""Cross-fitted validation-only check for the frozen support gate.

This is the topology gate without any validation-derived column calibration.
It accesses only cache['val'] and generator ``val`` logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support_gate(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    support = torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj]))
    return (support - support.min()) / (support.max() - support.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.2)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Deliberately do not access cache['test'].
    val = cache["val"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    scores = [zscore(item["val"].float()) for item in saved]
    gate = support_gate(cache)
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    n = len(val["pair"])
    folds = torch.arange(n) % args.folds
    rows = []
    for seed_id, score in enumerate(scores):
        for fold in range(args.folds):
            hold = folds == fold
            base = evaluator(val["logits"][hold].float(), val["attr"][hold], val["obj"][hold], val["pair"][hold])
            fused = evaluator(
                val["logits"][hold].float() + args.strength * score[hold] * gate[None],
                val["attr"][hold], val["obj"][hold], val["pair"][hold],
            )
            rows.append({
                "generator_seed_index": seed_id,
                "fold": fold,
                "holdout_count": int(hold.sum()),
                "auc_gain": fused["AUC"] - base["AUC"],
                "hm_gain": fused["best_hm"] - base["best_hm"],
                "base_val": base,
                "fused_val": fused,
            })
    result = {
        "protocol": {
            "mode": "cross_fitted_validation_only",
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "candidate": "min_image_support_continuous_full",
            "strength": args.strength,
            "folds": args.folds,
            "test_read": False,
            "classifier_logits": [
                {"path": str(Path(p).resolve()), "sha256": sha256(Path(p).resolve())}
                for p in args.classifier_logits
            ],
        },
        "rows": rows,
        "mean": {
            "auc_gain": float(np.mean([r["auc_gain"] for r in rows])),
            "hm_gain": float(np.mean([r["hm_gain"] for r in rows])),
            "positive_auc_runs": int(sum(r["auc_gain"] > 0 for r in rows)),
            "positive_hm_runs": int(sum(r["hm_gain"] > 0 for r in rows)),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
