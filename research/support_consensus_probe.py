"""Validation-only probe for support-aware source-consensus fusion.

The candidate combines source agreement with seen-graph image support.  It
deliberately reads only cache['train'] and cache['val']; no test tensor is
loaded or inspected.  The grid is fixed in the script before evaluation.
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


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def support_gate(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    attr = attr_count[pair_attr]
    obj = obj_count[pair_obj]
    if mode == "min_image_support":
        raw = torch.minimum(attr, obj)
    elif mode == "harmonic_image_support":
        raw = 2.0 * attr * obj / (attr + obj).clamp_min(1e-6)
    else:
        raise ValueError(mode)
    return normalize(torch.log1p(raw))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Keep this probe validation-only: never access cache['test'].
    val = cache["val"]
    source_paths = [Path(path).resolve() for path in args.logits]
    sources = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in source_paths
    ])
    median = sources.median(0).values
    dispersion = (sources - median).abs().mean(0)
    dispersion_scale = dispersion.median(1, keepdim=True).values.clamp_min(1e-6)
    relative_dispersion = dispersion / dispersion_scale
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_delta = median - text_logits

    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    # This is the complete predeclared search grid.
    support_modes = ("min_image_support", "harmonic_image_support")
    consensus_modes = (("none", 0.0), ("inverse", 0.5), ("inverse", 1.0),
                       ("inverse", 2.0), ("exp", 0.5), ("exp", 1.0), ("exp", 2.0))
    support_exponents = (0.0, 0.25, 0.5, 1.0)
    blend_modes = ("geometric", "linear")
    strengths = (0.5, 0.75, 1.0, 1.25)
    for support_mode in support_modes:
        support = support_gate(cache, support_mode)
        for consensus_name, tau in consensus_modes:
            if consensus_name == "none":
                agreement = torch.ones_like(relative_dispersion)
            elif consensus_name == "inverse":
                agreement = 1.0 / (1.0 + tau * relative_dispersion)
            else:
                agreement = torch.exp(-tau * relative_dispersion)
            agreement = agreement / agreement.mean(1, keepdim=True).clamp_min(1e-6)
            for exponent in support_exponents:
                support_power = support.clamp_min(1e-6).pow(exponent)
                for blend in blend_modes:
                    if blend == "geometric":
                        gate = agreement * support_power[None]
                    else:
                        gate = 0.5 * agreement + 0.5 * support_power[None]
                    gate = gate / gate.mean(1, keepdim=True).clamp_min(1e-6)
                    for strength in strengths:
                        fused = evaluator(
                            val["logits"].float() + strength * gate * base_delta,
                            val["attr"], val["obj"], val["pair"],
                        )
                        rows.append({
                            "support_mode": support_mode,
                            "consensus": consensus_name,
                            "tau": tau,
                            "support_exponent": exponent,
                            "blend": blend,
                            "strength": strength,
                            "auc_gain": fused["AUC"] - base["AUC"],
                            "hm_gain": fused["best_hm"] - base["best_hm"],
                            "fused_val": fused,
                        })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    best_hm = max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"]))
    result = {
        "mode": "validation_only_support_aware_consensus",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "base_val": base,
        "candidate_grid": {
            "support_modes": list(support_modes),
            "consensus_modes": [[name, tau] for name, tau in consensus_modes],
            "support_exponents": list(support_exponents),
            "blend_modes": list(blend_modes),
            "strengths": list(strengths),
            "selection": "among AUC gain >= 0.002, maximize validation HM",
        },
        "best_auc": best_auc,
        "best_balanced": best_hm,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "base_val": base,
        "best_auc": {k: best_auc[k] for k in ("auc_gain", "hm_gain", "support_mode", "consensus", "tau", "support_exponent", "blend", "strength")},
        "best_balanced": None if best_hm is None else {k: best_hm[k] for k in ("auc_gain", "hm_gain", "support_mode", "consensus", "tau", "support_exponent", "blend", "strength")},
    }, indent=2))


if __name__ == "__main__":
    main()
