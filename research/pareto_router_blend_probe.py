"""Validation-only blend of two already-frozen routing candidates."""

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


def normalized_pair_support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    raw = torch.log1p(torch.minimum(ac[pair_attr], oc[pair_obj]))
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


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
    support = normalized_pair_support(cache)
    inverse_score = (1.0 - support).clamp_min(1e-6).pow(1.0)[None] * diag + (1.0 - (1.0 - support).clamp_min(1e-6).pow(1.0)[None]) * iso
    soft_score = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25)[None]) * iso
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    soft_scope = unseen + (1.0 - unseen) * 0.25
    delta_a = inverse_score - text_logits
    delta_b = soft_score * soft_scope[None] * 1.5
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for blend_b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        delta = (1.0 - blend_b) * delta_a + blend_b * delta_b
        fused = evaluator(val["logits"].float() + delta, val["attr"], val["obj"], val["pair"])
        rows.append({"blend_soft": blend_b, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    result = {"mode": "validation_only_pareto_router_blend", "test_read": False, "cache": {"path": str(cache_path), "sha256": sha256(cache_path)}, "sources": [{"path": str(path), "sha256": sha256(path)} for path in iso_paths + diag_paths], "candidates": {"A": "inverse_pair_support^1/update x 1.0", "B": "pair_support^0.25/soft seen_lambda .25/full x 1.5"}, "blend_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "selection": "retain AUC gain >= .002 and maximize HM", "base_val": base, "best_balanced": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None, "best_auc": max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"])), "rows": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": result["best_balanced"], "best_auc": result["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
