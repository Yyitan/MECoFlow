"""Deterministic controls for the primitive-conditioned feature-flow pilot.

The script consumes the feature cache produced by piflow_pilot.py and evaluates
simple missing-cell completion rules. These controls separate gains from stochastic
distribution learning from gains obtainable by additive primitive prototypes alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test as cams_test  # noqa: E402
from dataset import CompositionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(dim=1, keepdim=True)) / logits.std(dim=1, keepdim=True).clamp_min(1e-6)


def stats_for(dataset: CompositionDataset, split: dict[str, torch.Tensor], logits: torch.Tensor) -> dict[str, float]:
    evaluator = cams_test.Evaluator(dataset)
    raw = cams_test.test(
        dataset,
        evaluator,
        logits,
        split["attr"],
        split["obj"],
        split["pair"],
    )
    return {key: float(raw[key]) for key in ("best_seen", "best_unseen", "best_hm", "AUC")}


def real_centroids(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    train = cache["train"]
    pair_count = len(cache["meta"]["pairs"])
    feature_dim = train["com"].shape[1]
    centroids = torch.zeros(pair_count, feature_dim)
    counts = torch.zeros(pair_count)
    for pair_id in torch.unique(train["pair"]):
        mask = train["pair"] == pair_id
        centroids[pair_id] = train["com"][mask].mean(dim=0)
        counts[pair_id] = mask.sum()
    return F.normalize(centroids, dim=-1), counts


def additive_completion(
    cache: dict[str, object], centroids: torch.Tensor, counts: torch.Tensor, iterations: int
) -> torch.Tensor:
    meta = cache["meta"]
    num_attrs, num_objs = len(meta["attrs"]), len(meta["objs"])
    pair_attr = torch.tensor([meta["attrs"].index(a) for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([meta["objs"].index(o) for _, o in meta["pairs"]])
    seen = counts > 0
    weights = counts[seen, None]
    global_mean = (centroids[seen] * weights).sum(dim=0) / weights.sum()
    attr_effect = torch.zeros(num_attrs, centroids.shape[1])
    obj_effect = torch.zeros(num_objs, centroids.shape[1])
    for _ in range(iterations):
        for attr_id in range(num_attrs):
            mask = seen & (pair_attr == attr_id)
            residual = centroids[mask] - global_mean - obj_effect[pair_obj[mask]]
            attr_effect[attr_id] = (residual * counts[mask, None]).sum(dim=0) / counts[mask].sum().clamp_min(1)
        attr_effect -= attr_effect.mean(dim=0, keepdim=True)
        for obj_id in range(num_objs):
            mask = seen & (pair_obj == obj_id)
            residual = centroids[mask] - global_mean - attr_effect[pair_attr[mask]]
            obj_effect[obj_id] = (residual * counts[mask, None]).sum(dim=0) / counts[mask].sum().clamp_min(1)
        obj_effect -= obj_effect.mean(dim=0, keepdim=True)
    prediction = global_mean + attr_effect[pair_attr] + obj_effect[pair_obj]
    return F.normalize(prediction, dim=-1)


def tune_fusion(
    val_dataset: CompositionDataset,
    split: dict[str, torch.Tensor],
    prototype_logits: torch.Tensor,
) -> tuple[float, dict[str, float]]:
    best_weight, best_stats = 0.0, None
    for weight in np.linspace(0.0, 1.0, 21):
        logits = (1.0 - weight) * zscore(split["logits"]) + weight * zscore(prototype_logits)
        stats = stats_for(val_dataset, split, logits)
        if best_stats is None or (stats["AUC"], stats["best_hm"]) > (best_stats["AUC"], best_stats["best_hm"]):
            best_weight, best_stats = float(weight), stats
    assert best_stats is not None
    return best_weight, best_stats


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, args.iterations)
    hybrid = additive.clone()
    hybrid[counts > 0] = centroids[counts > 0]
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    variants = {"text": text, "additive": additive, "additive_hybrid": hybrid}

    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    result: dict[str, object] = {
        "base_val": stats_for(val_dataset, cache["val"], cache["val"]["logits"]),
        "base_test": stats_for(test_dataset, cache["test"], cache["test"]["logits"]),
    }
    for name, prototypes in variants.items():
        val_proto = cache["val"]["com"] @ prototypes.T
        test_proto = cache["test"]["com"] @ prototypes.T
        weight, val_fused = tune_fusion(val_dataset, cache["val"], val_proto)
        test_fused = (1.0 - weight) * zscore(cache["test"]["logits"]) + weight * zscore(test_proto)
        result[name] = {
            "only_val": stats_for(val_dataset, cache["val"], val_proto),
            "only_test": stats_for(test_dataset, cache["test"], test_proto),
            "selected_weight": weight,
            "fused_val": val_fused,
            "fused_test": stats_for(test_dataset, cache["test"], test_fused),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
