"""Cross-fitted validation-only check for a fixed residual-source ensemble."""

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
    parser.add_argument("--diag-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--strength", type=float, default=1.25)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Validation-only by construction.
    val = cache["val"]
    diag = [zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in args.diag_logits]
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    delta = torch.stack(diag).mean(0) - zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    folds = torch.arange(len(val["pair"])) % args.folds
    rows = []
    for fold in range(args.folds):
        hold = folds == fold
        base = evaluator(val["logits"][hold].float(), val["attr"][hold], val["obj"][hold], val["pair"][hold])
        fused = evaluator(val["logits"][hold].float() + args.strength * delta[hold], val["attr"][hold], val["obj"][hold], val["pair"][hold])
        rows.append({
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
            "candidate": "diagonal_source_mean_update",
            "strength": args.strength,
            "folds": args.folds,
            "test_read": False,
            "diag_logits": [{"path": str(Path(p).resolve()), "sha256": sha256(Path(p).resolve())} for p in args.diag_logits],
        },
        "rows": rows,
        "mean": {
            "auc_gain": float(np.mean([r["auc_gain"] for r in rows])),
            "hm_gain": float(np.mean([r["hm_gain"] for r in rows])),
            "positive_auc_runs": int(sum(r["auc_gain"] > 0 for r in rows)),
            "positive_hm_runs": int(sum(r["hm_gain"] > 0 for r in rows)),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
