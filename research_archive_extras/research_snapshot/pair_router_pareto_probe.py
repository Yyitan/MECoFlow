"""Validation-only convex mixtures of pair-level source-routing priors."""

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


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def route_weights(cache: dict[str, object]) -> dict[str, torch.Tensor]:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    pair_raw = normalize(torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj])))
    obj_raw = normalize(torch.log1p(obj_count[pair_obj]))
    return {
        "inverse_pair_support": 1.0 - pair_raw,
        "obj_support": obj_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--iso-logits", nargs=3, required=True)
    parser.add_argument("--diag-logits", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    iso_paths = [Path(path).resolve() for path in args.iso_logits]
    diag_paths = [Path(path).resolve() for path in args.diag_logits]
    iso = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in iso_paths]).mean(0)
    diag = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in diag_paths]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    routes = route_weights(cache)
    rows = []
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        weights = mix * routes["inverse_pair_support"] + (1.0 - mix) * routes["obj_support"]
        score = weights[None] * diag + (1.0 - weights[None]) * iso
        for delta_name, delta in (("full", score), ("update", score - val_text)):
            for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
                fused = evaluator(val["logits"].float() + strength * delta, val["attr"], val["obj"], val["pair"])
                rows.append({
                    "inverse_pair_mix": mix,
                    "delta": delta_name,
                    "strength": strength,
                    "auc_gain": fused["AUC"] - base["AUC"],
                    "hm_gain": fused["best_hm"] - base["best_hm"],
                    "fused_val": fused,
                })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002 and row["hm_gain"] >= 0.010]
    best_auc_first = max(eligible, key=lambda row: (row["auc_gain"], row["hm_gain"])) if eligible else None
    best_hm_first = max([row for row in rows if row["auc_gain"] >= 0.002], key=lambda row: (row["hm_gain"], row["auc_gain"])) if any(row["auc_gain"] >= 0.002 for row in rows) else None
    result = {
        "mode": "validation_only_pair_router_pareto_mixture",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in iso_paths + diag_paths],
        "candidate_grid": {"inverse_pair_mix": [0.0, 0.25, 0.5, 0.75, 1.0], "delta": ["full", "update"], "strength": [0.5, 0.75, 1.0, 1.25, 1.5]},
        "selection_rules": {"auc_first": "AUC gain >= .002 and HM gain >= .010, maximize AUC", "hm_first": "AUC gain >= .002, maximize HM"},
        "base_val": base,
        "best_auc_first": best_auc_first,
        "best_hm_first": best_hm_first,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_auc_first": best_auc_first, "best_hm_first": best_hm_first}, indent=2))


if __name__ == "__main__":
    main()
