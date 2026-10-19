"""Validation-only classwise uncertainty calibration on a locked source posterior."""

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_norm(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def source_posterior(scores: torch.Tensor, center: str, tau: float, conf_name: str, conf_power: float) -> torch.Tensor:
    c = scores.median(0).values if center == "median" else scores.mean(0)
    distance = (scores - c[None]).abs().mean(2)
    distance = distance / distance.mean(1, keepdim=True).clamp_min(1e-6)
    top2 = torch.topk(scores, 2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    prob = torch.softmax(scores, dim=2)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(2)
    confidence = row_norm(margin) if conf_name == "margin_high" else 1.0 - row_norm(entropy)
    weights = torch.exp(-tau * distance) * confidence.clamp_min(1e-6).pow(conf_power)
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    return (weights[:, :, None] * scores).sum(0)


def class_gate(scores: torch.Tensor, mode: str, tau: float, power: float) -> torch.Tensor:
    uncertainty = scores.std(0, unbiased=False)
    relative = uncertainty / uncertainty.mean(1, keepdim=True).clamp_min(1e-6)
    if mode == "inverse":
        gate = 1.0 / (1.0 + tau * relative)
    elif mode == "exp":
        gate = torch.exp(-tau * relative)
    elif mode == "none":
        gate = torch.ones_like(relative)
    else:
        raise ValueError(mode)
    gate = gate.clamp_min(1e-6).pow(power)
    return gate / gate.mean(1, keepdim=True).clamp_min(1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--logits", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--posterior-center", choices=("median", "mean"), required=True)
    ap.add_argument("--posterior-confidence", choices=("margin_high", "entropy_low"), required=True)
    ap.add_argument("--posterior-tau", type=float, required=True)
    ap.add_argument("--posterior-power", type=float, required=True)
    args = ap.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Validation-only: no cache['test'] and no source['test'] access.
    val = cache["val"]
    paths = [Path(x).resolve() for x in args.logits]
    scores = torch.stack([zscore(torch.load(x, map_location="cpu", weights_only=True)["val"].float()) for x in paths])
    source = source_posterior(scores, args.posterior_center, args.posterior_tau, args.posterior_confidence, args.posterior_power)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_delta = source - text_logits
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for mode in ("none", "inverse", "exp"):
        for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
            for power in (0.25, 0.5, 1.0):
                gate = class_gate(scores, mode, tau, power)
                for delta_name, delta in (("full", source), ("update", base_delta)):
                    for strength in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
                        fused = evaluator(val["logits"].float() + strength * gate * delta, val["attr"], val["obj"], val["pair"])
                        rows.append({"class_mode": mode, "class_tau": tau, "class_power": power, "delta": delta_name, "strength": strength, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"]))
    result = {"mode": "validation_only_classwise_uncertainty_posterior", "test_read": False, "cache": {"path": str(cache_path), "sha256": sha256(cache_path)}, "sources": [{"path": str(x), "sha256": sha256(x)} for x in paths], "locked_source_posterior": {"center": args.posterior_center, "confidence": args.posterior_confidence, "tau": args.posterior_tau, "power": args.posterior_power}, "base_val": base, "candidate_grid": {"class_mode": ["none", "inverse", "exp"], "class_tau": [0.25, 0.5, 1.0, 2.0, 4.0], "class_power": [0.25, 0.5, 1.0], "delta": ["full", "update"], "strength": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5], "selection": "retain validation AUC gain >= 0.002, maximize validation HM"}, "best_balanced": best_balanced, "best_auc": best_auc, "rows": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": best_balanced, "best_auc": best_auc}, indent=2))


if __name__ == "__main__":
    main()
