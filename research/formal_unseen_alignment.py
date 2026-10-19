"""Measure why formal unseen pairs differ from random held-out train edges."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.fast_feature_evaluation import additive_completion, real_centroids  # noqa: E402
from research.warmstart_flow_classifier import pair_maps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-8)
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-8)
    return (x * y).mean().item()


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    prototypes = torch.load(args.prototypes, map_location="cpu", weights_only=True).float()
    train_centroids, train_counts = real_centroids(cache)
    additive = additive_completion(cache, train_centroids, train_counts, 30)
    val = cache["val"]
    val_ids = torch.unique(val["pair"].long()).sort().values
    real = []
    for pair_id in val_ids.tolist():
        real.append(F.normalize(val["com"][val["pair"] == pair_id].float().mean(0), dim=-1))
    real = torch.stack(real)
    generated_alignment = F.cosine_similarity(real, prototypes[val_ids], dim=-1)
    additive_alignment = F.cosine_similarity(real, additive[val_ids], dim=-1)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_alignment = F.cosine_similarity(real, text[val_ids], dim=-1)

    seen_ids = torch.where(train_counts > 0)[0]
    semantic_nearest = (text[val_ids] @ text[seen_ids].T).max(-1).values
    pair_attr, pair_obj = pair_maps(cache)
    train_pair = train_counts > 0
    attr_degree = torch.bincount(
        pair_attr[train_pair], minlength=len(cache["meta"]["attrs"])
    ).float()
    obj_degree = torch.bincount(
        pair_obj[train_pair], minlength=len(cache["meta"]["objs"])
    ).float()
    minimum_degree = torch.minimum(
        attr_degree[pair_attr[val_ids]], obj_degree[pair_obj[val_ids]]
    )
    result = {
        "formal_val_pairs": len(val_ids),
        "generated_alignment_mean": generated_alignment.mean().item(),
        "generated_alignment_median": generated_alignment.median().item(),
        "additive_alignment_mean": additive_alignment.mean().item(),
        "text_alignment_mean": text_alignment.mean().item(),
        "nearest_seen_semantic_mean": semantic_nearest.mean().item(),
        "minimum_primitive_degree_mean": minimum_degree.mean().item(),
        "corr_generated_semantic_nearest": correlation(
            generated_alignment, semantic_nearest
        ),
        "corr_generated_minimum_degree": correlation(
            generated_alignment, minimum_degree
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
