"""Cross-fitted validation-only check for pair-column calibrated score updates.

The candidate is fixed before this script runs: z-score each generator's pair
scores after subtracting a column mean fitted on the complementary validation
fold, then apply the fixed min-image-support gate at strength 0.20. No test
branch is accessed.
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
    support = torch.minimum(attr_count[pair_attr], obj_count[pair_obj])
    support = torch.log1p(support)
    return (support - support.min()) / (support.max() - support.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.20)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds must be >= 2")

    torch.set_num_threads(min(torch.get_num_threads(), 16))
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Deliberately access only the validation branch. Do not touch cache['test'].
    val = cache["val"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    scores = [zscore(item["val"].float()) for item in saved]
    gate = support_gate(cache)
    n = len(val["pair"])
    fold_ids = torch.arange(n) % args.folds
    dataset = CompositionDataset(args.dataset_path, phase="val")
    evaluator = FastClosedWorldEvaluator(dataset, torch.device("cpu"))
    rows: list[dict[str, object]] = []
    for seed_id, score in enumerate(scores):
        for fold in range(args.folds):
            hold = fold_ids == fold
            calibration = ~hold
            column_mean = score[calibration].mean(0, keepdim=True)
            calibrated = zscore(score[hold] - column_mean)
            base_stats = evaluator(
                val["logits"][hold].float(), val["attr"][hold],
                val["obj"][hold], val["pair"][hold],
            )
            fused_stats = evaluator(
                val["logits"][hold].float() + args.strength * calibrated * gate[None],
                val["attr"][hold], val["obj"][hold], val["pair"][hold],
            )
            rows.append({
                "generator_seed_index": seed_id,
                "fold": fold,
                "holdout_count": int(hold.sum()),
                "base_val": base_stats,
                "fused_val": fused_stats,
                "auc_gain": fused_stats["AUC"] - base_stats["AUC"],
                "hm_gain": fused_stats["best_hm"] - base_stats["best_hm"],
            })
    output = {
        "protocol": {
            "mode": "cross_fitted_validation_only",
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "candidate": "column_centered_generated_score",
            "gate": "min_image_support_continuous_full",
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
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["mean"], indent=2))


if __name__ == "__main__":
    main()
