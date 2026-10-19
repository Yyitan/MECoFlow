"""Validation-only replay of a previously locked adaptive source posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
    return (x - x.amin(1, keepdim=True)) / (x.amax(1, keepdim=True) - x.amin(1, keepdim=True)).clamp_min(1e-6)


def confidence(scores: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(scores, 2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    probs = torch.softmax(scores, dim=2)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(2)
    if name == "margin_high":
        return row_norm(margin)
    if name == "entropy_low":
        return 1 - row_norm(entropy)
    raise ValueError(name)


def posterior(scores: torch.Tensor, candidate: dict[str, object]) -> torch.Tensor:
    center = scores.median(0).values if candidate["center"] == "median" else scores.mean(0)
    residual = scores - center[None]
    distance = residual.abs().mean(2) if candidate["distance"] == "abs" else residual.square().mean(2).sqrt()
    distance = distance / distance.mean(1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-float(candidate["tau"]) * distance)
    weights = weights * confidence(scores, candidate["confidence"]).clamp_min(1e-6).pow(float(candidate["confidence_power"]))
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    return (weights[:, :, None] * scores).sum(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--logits", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    paths = [Path(x).resolve() for x in args.logits]
    scores = torch.stack([zscore(torch.load(x, map_location="cpu", weights_only=True)["val"].float()) for x in paths])
    candidate = {"center": "mean", "distance": "abs", "confidence": "entropy_low", "tau": 4.0, "confidence_power": 1.0, "delta": "update", "strength": 0.75}
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    source = posterior(scores, candidate)
    delta = source - text_logits
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused = evaluator(val["logits"].float() + candidate["strength"] * delta, val["attr"], val["obj"], val["pair"])
    result = {"mode": "validation_only_locked_adaptive_replay", "test_read": False, "candidate": candidate, "cache": {"path": str(cache_path), "sha256": sha256(cache_path)}, "sources": [{"path": str(x), "sha256": sha256(x)} for x in paths], "base_val": base, "fused_val": fused, "mean": {"auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"]}}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
