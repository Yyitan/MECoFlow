"""Validation-only probe for factorized attribute/object support gates."""

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


def factor_support(cache: dict[str, object], alpha: float) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()[pair_attr]
    obj = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()[pair_obj]
    log_attr = torch.log1p(attr)
    log_obj = torch.log1p(obj)
    return normalize(alpha * log_attr + (1.0 - alpha) * log_obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Validation-only invariant: never touch cache['test'] or logits['test'].
    val = cache["val"]
    source_paths = [Path(p).resolve() for p in args.logits]
    scores = [zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in source_paths]
    stack = torch.stack(scores)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    # Entire grid is predeclared before reading validation outcomes.
    for aggregation in ("median", "mean"):
        source = stack.median(0).values if aggregation == "median" else stack.mean(0)
        delta = source - text_logits
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            support = factor_support(cache, alpha)
            for transform, parameter in (("power", 0.25), ("power", 0.5), ("power", 1.0),
                                         ("floor", 0.25), ("floor", 0.5), ("floor", 0.75)):
                if transform == "power":
                    gate = support.clamp_min(1e-6).pow(parameter)
                else:
                    gate = (1.0 - parameter) + parameter * support
                for strength in (0.25, 0.5, 0.75, 1.0):
                    sample_gate = gate[None].expand(delta.shape[0], -1)
                    sample_gate = sample_gate / sample_gate.mean(1, keepdim=True).clamp_min(1e-6)
                    fused = evaluator(val["logits"].float() + strength * sample_gate * delta,
                                      val["attr"], val["obj"], val["pair"])
                    rows.append({
                        "aggregation": aggregation,
                        "alpha_attr": alpha,
                        "transform": transform,
                        "parameter": parameter,
                        "strength": strength,
                        "auc_gain": fused["AUC"] - base["AUC"],
                        "hm_gain": fused["best_hm"] - base["best_hm"],
                        "fused_val": fused,
                    })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"]))
    result = {
        "mode": "validation_only_factorized_support",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "base_val": base,
        "candidate_grid": {
            "aggregation": ["median", "mean"],
            "alpha_attr": [0.0, 0.25, 0.5, 0.75, 1.0],
            "transforms": [["power", 0.25], ["power", 0.5], ["power", 1.0],
                            ["floor", 0.25], ["floor", 0.5], ["floor", 0.75]],
            "strength": [0.25, 0.5, 0.75, 1.0],
            "selection": "retain validation AUC gain >= 0.002 and maximize validation HM",
        },
        "best_balanced": best_balanced,
        "best_auc": best_auc,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_balanced": None if best_balanced is None else {k: best_balanced[k] for k in ("auc_gain", "hm_gain", "aggregation", "alpha_attr", "transform", "parameter", "strength")},
        "best_auc": {k: best_auc[k] for k in ("auc_gain", "hm_gain", "aggregation", "alpha_attr", "transform", "parameter", "strength")},
    }, indent=2))


if __name__ == "__main__":
    main()
