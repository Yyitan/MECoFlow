"""Validation-only mixture of locked diagonal-family and support residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402
from research.residual_mixture_probe import support_delta  # noqa: E402


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
    parser.add_argument("--isotropic", nargs=3, required=True)
    parser.add_argument("--diagonal", nargs=3, required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    iso_paths = [Path(v).resolve() for v in args.isotropic]
    diag_paths = [Path(v).resolve() for v in args.diagonal]
    iso = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in iso_paths])
    diag = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in diag_paths])
    diagonal_delta = diag.mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    support_cfg = json.loads(Path(args.support).read_text(encoding="utf-8"))
    all_scores = torch.cat([iso, diag], dim=0)
    support_delta_value, support_values = support_delta(all_scores, text_logits, cache, support_cfg)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    alphas = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    for alpha in alphas:
        delta = alpha * 1.25 * diagonal_delta + (1.0 - alpha) * 0.75 * support_delta_value
        fused = evaluator(val["logits"].float() + delta, val["attr"], val["obj"], val["pair"])
        rows.append({"alpha_diagonal": alpha, "alpha_support": 1.0 - alpha, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"]))
    result = {
        "mode": "validation_only_locked_diagonal_support_mixture",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in iso_paths + diag_paths],
        "base_val": base,
        "components": {"diagonal": {"aggregation": "mean_zscore", "strength": 1.25}, "support": {"config": support_cfg, "strength": 0.75}},
        "candidate_grid": {"alpha_diagonal": list(alphas), "selection": "retain AUC gain >= 0.002, maximize validation HM"},
        "best_balanced": best_balanced,
        "best_auc": best_auc,
        "rows": rows,
        "diagnostics": {"support_gate_mean": support_values.mean(1).tolist()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": best_balanced, "best_auc": best_auc}, indent=2))


if __name__ == "__main__":
    main()
