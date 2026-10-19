"""Validation-only probe for source-consensus gated residual fusion."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # This probe deliberately loads cache['val'] only; test is never touched.
    val = cache["val"]
    source_paths = [Path(p).resolve() for p in args.logits]
    sources = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in source_paths
    ])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    median = sources.median(0).values
    dispersion = (sources - median).abs().mean(0)
    scale = dispersion.median(1, keepdim=True).values.clamp_min(1e-6)
    normalized_dispersion = dispersion / scale

    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    candidates = [("none", 0.0), ("exp", 0.25), ("exp", 0.5), ("exp", 1.0), ("exp", 2.0),
                  ("inverse", 0.25), ("inverse", 0.5), ("inverse", 1.0), ("inverse", 2.0)]
    for gate_name, tau in candidates:
        if gate_name == "none":
            gate = torch.ones_like(normalized_dispersion)
        elif gate_name == "exp":
            gate = torch.exp(-tau * normalized_dispersion)
        else:
            gate = 1.0 / (1.0 + tau * normalized_dispersion)
        # Preserve each sample's average update magnitude while suppressing
        # classes whose source estimates disagree unusually strongly.
        gate = gate / gate.mean(1, keepdim=True).clamp_min(1e-6)
        delta = gate * (median - text_logits)
        for strength in (0.75, 1.0, 1.25):
            fused = evaluator(
                val["logits"].float() + strength * delta,
                val["attr"], val["obj"], val["pair"]
            )
            rows.append({
                "gate": gate_name,
                "tau": tau,
                "strength": strength,
                "auc_gain": fused["AUC"] - base["AUC"],
                "hm_gain": fused["best_hm"] - base["best_hm"],
                "fused_val": fused,
            })
    rows.sort(key=lambda row: (row["auc_gain"], row["hm_gain"]), reverse=True)
    result = {
        "mode": "validation_only_consensus_gate",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "base_val": base,
        "rows": rows,
        "best": rows[0],
        "candidate_grid": {"gates": ["none", "exp", "inverse"], "tau": [0.25, 0.5, 1.0, 2.0], "strength": [0.75, 1.0, 1.25]},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"base_val": base, "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
