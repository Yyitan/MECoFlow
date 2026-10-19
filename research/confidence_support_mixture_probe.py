"""Validation-only mixture of locked confidence and support residual experts."""

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
from research.residual_mixture_probe import support_delta, support_gate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def confidence_gate(logits: torch.Tensor, name: str, power: float) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
    if name == "base_entropy_low":
        values = 1.0 - normalize(entropy)
    elif name == "base_entropy_high":
        values = normalize(entropy)
    elif name == "base_margin_high":
        values = normalize(margin)
    else:
        raise ValueError(name)
    gate = values.clamp_min(1e-6).pow(power)
    return gate / gate.mean().clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", required=True)
    parser.add_argument("--confidence-power", type=float, required=True)
    parser.add_argument("--confidence-strength", type=float, required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--support-strength", type=float, required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    source_paths = [Path(v).resolve() for v in args.logits]
    scores = torch.stack([
        zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float())
        for p in source_paths
    ])
    source = scores.median(0).values
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_delta = source - text_logits
    confidence = confidence_gate(zscore(val["logits"].float()), args.confidence, args.confidence_power)
    confidence_delta = confidence[:, None] * base_delta
    support_cfg = json.loads(Path(args.support).read_text(encoding="utf-8"))
    support_delta_value, support_values = support_delta(scores, text_logits, cache, support_cfg)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    alphas = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    for alpha in alphas:
        delta = (alpha * float(args.confidence_strength) * confidence_delta
                 + (1.0 - alpha) * float(args.support_strength) * support_delta_value)
        fused = evaluator(val["logits"].float() + delta, val["attr"], val["obj"], val["pair"])
        rows.append({
            "alpha_confidence": alpha,
            "alpha_support": 1.0 - alpha,
            "auc_gain": fused["AUC"] - base["AUC"],
            "hm_gain": fused["best_hm"] - base["best_hm"],
            "fused_val": fused,
        })
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"]))
    result = {
        "mode": "validation_only_locked_confidence_support_mixture",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in source_paths],
        "base_val": base,
        "components": {
            "confidence": {"name": args.confidence, "power": args.confidence_power, "strength": args.confidence_strength},
            "support": {"config": support_cfg, "strength": args.support_strength},
        },
        "candidate_grid": {"alpha_confidence": list(alphas), "selection": "retain AUC gain >= 0.002, maximize validation HM"},
        "best_balanced": best_balanced,
        "best_auc": best_auc,
        "rows": rows,
        "diagnostics": {"support_gate_mean": support_values.mean(1).tolist()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_balanced": None if best_balanced is None else {k: best_balanced[k] for k in ("auc_gain", "hm_gain", "alpha_confidence", "alpha_support")},
        "best_auc": {k: best_auc[k] for k in ("auc_gain", "hm_gain", "alpha_confidence", "alpha_support")},
    }, indent=2))


if __name__ == "__main__":
    main()
