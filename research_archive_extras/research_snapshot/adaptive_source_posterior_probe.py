"""Validation-only non-parametric posterior over source generators.

Each image gets source weights from two observable signals only: distance to a
robust source center and the source's own class confidence.  Labels are used
only by the outer validation gate, never by the per-image weights.
"""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_norm(value: torch.Tensor) -> torch.Tensor:
    lo = value.amin(dim=1, keepdim=True)
    hi = value.amax(dim=1, keepdim=True)
    return (value - lo) / (hi - lo).clamp_min(1e-6)


def confidence(scores: torch.Tensor, mode: str) -> torch.Tensor:
    top2 = torch.topk(scores, k=2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    probs = torch.softmax(scores, dim=2)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(2)
    if mode == "none":
        return torch.ones(scores.shape[:2])
    if mode == "margin_high":
        return row_norm(margin)
    if mode == "entropy_low":
        return 1.0 - row_norm(entropy)
    raise ValueError(mode)


def posterior(scores: torch.Tensor, center_name: str, distance_name: str,
              tau: float, confidence_name: str, confidence_power: float) -> tuple[torch.Tensor, torch.Tensor]:
    center = scores.median(0).values if center_name == "median" else scores.mean(0)
    residual = scores - center[None]
    if distance_name == "abs":
        distance = residual.abs().mean(2)
    elif distance_name == "rms":
        distance = residual.square().mean(2).sqrt()
    else:
        raise ValueError(distance_name)
    distance = distance / distance.mean(1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-tau * distance)
    weights = weights * confidence(scores, confidence_name).clamp_min(1e-6).pow(confidence_power)
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    fused = (weights[:, :, None] * scores).sum(0)
    return fused, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Validation-only: no cache['test'] or source['test'] access.
    val = cache["val"]
    source_paths = [Path(value).resolve() for value in args.logits]
    scores = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in source_paths
    ])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    centers = ("median", "mean")
    distances = ("abs", "rms")
    confidence_modes = ("none", "margin_high", "entropy_low")
    taus = (0.25, 0.5, 1.0, 2.0, 4.0)
    powers = (0.0, 0.25, 0.5, 1.0)
    strengths = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    for center_name in centers:
        for distance_name in distances:
            for confidence_name in confidence_modes:
                for tau in taus:
                    for confidence_power in powers:
                        score, weights = posterior(scores, center_name, distance_name, tau, confidence_name, confidence_power)
                        for delta_name, delta in (("full", score), ("update", score - text_logits)):
                            for strength in strengths:
                                fused = evaluator(val["logits"].float() + strength * delta,
                                                   val["attr"], val["obj"], val["pair"])
                                rows.append({
                                    "center": center_name,
                                    "distance": distance_name,
                                    "confidence": confidence_name,
                                    "tau": tau,
                                    "confidence_power": confidence_power,
                                    "delta": delta_name,
                                    "strength": strength,
                                    "auc_gain": fused["AUC"] - base["AUC"],
                                    "hm_gain": fused["best_hm"] - base["best_hm"],
                                    "weight_mean": weights.mean(1).tolist(),
                                    "fused_val": fused,
                                })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"]))
    result = {
        "mode": "validation_only_adaptive_source_posterior",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "base_val": base,
        "candidate_grid": {
            "center": list(centers), "distance": list(distances),
            "confidence": list(confidence_modes), "tau": list(taus),
            "confidence_power": list(powers), "delta": ["full", "update"],
            "strength": list(strengths),
            "selection": "retain validation AUC gain >= 0.002, maximize validation HM",
        },
        "best_balanced": best_balanced,
        "best_auc": best_auc,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": best_balanced, "best_auc": best_auc}, indent=2))


if __name__ == "__main__":
    main()
