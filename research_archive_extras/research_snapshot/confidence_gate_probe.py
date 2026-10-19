"""Validation-only probe for confidence-aware transport updates."""

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


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Strict validation-only invariant.
    val = cache["val"]
    paths = [Path(path).resolve() for path in args.logits]
    scores = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in paths])
    source = scores.median(0).values
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_logits = zscore(val["logits"].float())
    delta = source - text_logits
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])

    # Image-level confidence summaries expanded to class columns.
    def margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        return -(probs * probs.clamp_min(1e-8).log()).sum(1)

    uncertainty = {
        "text_margin_low": 1.0 - normalize(margin(text_logits)),
        "text_margin_high": normalize(margin(text_logits)),
        "base_margin_low": 1.0 - normalize(margin(base_logits)),
        "base_margin_high": normalize(margin(base_logits)),
        "text_entropy_high": normalize(entropy(text_logits)),
        "text_entropy_low": 1.0 - normalize(entropy(text_logits)),
        "base_entropy_high": normalize(entropy(base_logits)),
        "base_entropy_low": 1.0 - normalize(entropy(base_logits)),
    }
    rows = []
    # Gate is applied per image (same scalar for all classes); normalization
    # preserves the mean update magnitude.
    for name, values in uncertainty.items():
        for power in (0.25, 0.5, 1.0, 2.0):
            gate = values.clamp_min(1e-6).pow(power)
            gate = gate / gate.mean().clamp_min(1e-6)
            for strength in (0.25, 0.5, 0.75, 1.0, 1.25):
                fused = evaluator(val["logits"].float() + strength * gate[:, None] * delta,
                                  val["attr"], val["obj"], val["pair"])
                rows.append({
                    "confidence": name,
                    "power": power,
                    "strength": strength,
                    "auc_gain": fused["AUC"] - base["AUC"],
                    "hm_gain": fused["best_hm"] - base["best_hm"],
                    "fused_val": fused,
                })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    result = {
        "mode": "validation_only_confidence_gate",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "base_val": base,
        "candidate_grid": {
            "confidence": list(uncertainty),
            "power": [0.25, 0.5, 1.0, 2.0],
            "strength": [0.25, 0.5, 0.75, 1.0, 1.25],
            "selection": "retain validation AUC gain >= 0.002 and maximize validation HM",
        },
        "best_balanced": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None,
        "best_auc": max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"])),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_balanced": result["best_balanced"],
        "best_auc": result["best_auc"],
    }, indent=2))


if __name__ == "__main__":
    main()
