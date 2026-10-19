"""Validation-only support-tapered selective pair routing."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def route_weight(cache: dict[str, object], route: str, power: float) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    if route in {"pair_support", "inverse_pair_support"}:
        raw = torch.log1p(torch.minimum(ac[pair_attr], oc[pair_obj]))
    elif route in {"support_balance", "inverse_support_balance"}:
        raw = torch.minimum(ac[pair_attr], oc[pair_obj]) / torch.maximum(ac[pair_attr], oc[pair_obj]).clamp_min(1.0)
    elif route == "obj_support":
        raw = torch.log1p(oc[pair_obj])
    else:
        raise ValueError(route)
    weight = normalize(raw)
    if route in {"inverse_pair_support", "inverse_support_balance"}:
        weight = 1.0 - weight
    return weight.clamp_min(1e-6).pow(power)


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
    text_logits = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    rows = []
    for route_name in ("pair_support", "inverse_pair_support", "support_balance", "inverse_support_balance", "obj_support"):
        for power in (0.25, 0.5, 1.0):
            weight = route_weight(cache, route_name, power)
            score = weight[None] * diag + (1.0 - weight[None]) * iso
            for seen_lambda in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
                scope = unseen + (1.0 - unseen) * seen_lambda
                for delta_name, delta in (("full", score), ("update", score - text_logits)):
                    for strength in (-1.5, -1.25, -1.0, -0.75, -0.5, 0.5, 0.75, 1.0, 1.25, 1.5):
                        fused = evaluator(val["logits"].float() + strength * delta * scope[None], val["attr"], val["obj"], val["pair"])
                        rows.append({"route": route_name, "power": power, "seen_lambda": seen_lambda, "delta": delta_name, "strength": strength, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    result = {"mode": "validation_only_soft_selective_pair_router", "test_read": False, "cache": {"path": str(cache_path), "sha256": sha256(cache_path)}, "sources": [{"path": str(path), "sha256": sha256(path)} for path in iso_paths + diag_paths], "candidate_grid": {"routes": ["pair_support", "inverse_pair_support", "obj_support"], "powers": [0.25, 0.5, 1.0], "seen_lambda": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0], "delta": ["full", "update"], "strength": [0.5, 0.75, 1.0, 1.25, 1.5]}, "selection": "retain AUC gain >= .002 and maximize HM", "base_val": base, "best_balanced": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None, "best_auc": max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"]))}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": result["best_balanced"], "best_auc": result["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
