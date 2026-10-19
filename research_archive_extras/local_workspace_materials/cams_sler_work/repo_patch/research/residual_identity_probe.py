"""Measure how much seen-pair identity remains in composition residuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--evaluator-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-probe-classes", type=int, default=0)
    return parser.parse_args()


def stratified_half(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_ids, test_ids = [], []
    for label in torch.unique(labels).tolist():
        ids = torch.where(labels == label)[0]
        ids = ids[torch.randperm(len(ids), generator=generator)]
        cut = max(1, len(ids) // 2)
        train_ids.append(ids[:cut])
        test_ids.append(ids[cut:])
    return torch.cat(train_ids), torch.cat(test_ids)


def centroid_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_ids: torch.Tensor,
    test_ids: torch.Tensor,
    class_count: int,
) -> float:
    global_mean = features[train_ids].mean(0, keepdim=True)
    train = F.normalize(features[train_ids] - global_mean, dim=-1)
    test = F.normalize(features[test_ids] - global_mean, dim=-1)
    centroids = features.new_zeros(class_count, features.shape[1])
    centroids.index_add_(0, labels[train_ids], train)
    counts = torch.bincount(labels[train_ids], minlength=class_count).clamp_min(1)
    centroids = F.normalize(centroids / counts[:, None], dim=-1)
    prediction = (test @ centroids.T).argmax(1)
    return float((prediction == labels[test_ids]).float().mean())


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.evaluator_repo).resolve()))
    from research.fast_feature_evaluation import (  # pylint: disable=import-outside-toplevel
        additive_completion,
        real_centroids,
    )

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    centroids, counts = real_centroids(cache)
    additive = additive_completion(cache, centroids, counts, 30)
    labels = cache["train"]["pair"].long()
    seen_ids = torch.unique(labels).sort().values
    compact = torch.full((len(counts),), -1, dtype=torch.long)
    compact[seen_ids] = torch.arange(len(seen_ids))
    compact_labels = compact[labels]
    residuals = cache["train"]["com"].float() - additive[labels]
    full_class_count = len(seen_ids)
    if args.max_probe_classes > 0 and full_class_count > args.max_probe_classes:
        generator = torch.Generator().manual_seed(args.seed + 104729)
        selected = torch.randperm(full_class_count, generator=generator)[
            : args.max_probe_classes
        ].sort().values
        selected_mask = torch.zeros(full_class_count, dtype=torch.bool)
        selected_mask[selected] = True
        sample_mask = selected_mask[compact_labels]
        residuals = residuals[sample_mask]
        selected_labels = compact_labels[sample_mask]
        remap = torch.full((full_class_count,), -1, dtype=torch.long)
        remap[selected] = torch.arange(len(selected))
        compact_labels = remap[selected_labels]
    train_ids, test_ids = stratified_half(compact_labels, args.seed)
    class_count = int(compact_labels.max()) + 1

    mean = residuals.mean(0)
    diag_std = residuals.std(0, unbiased=False).clamp_min(1e-6)
    iso_std = diag_std.square().mean().sqrt()
    axis_variance = diag_std.square()
    variance_fraction = axis_variance / axis_variance.sum().clamp_min(1e-12)
    axis_participation_rank = (
        axis_variance.sum().square() / axis_variance.square().sum().clamp_min(1e-12)
    )
    axis_entropy_rank = torch.exp(
        -(variance_fraction * variance_fraction.clamp_min(1e-12).log()).sum()
    )
    generator = torch.Generator().manual_seed(args.seed + 7919)
    diag = mean + torch.randn(residuals.shape, generator=generator) * diag_std
    iso = mean + torch.randn(residuals.shape, generator=generator) * iso_std

    pair_means = residuals.new_zeros(class_count, residuals.shape[1])
    pair_means.index_add_(0, compact_labels, residuals)
    pair_counts = torch.bincount(compact_labels, minlength=class_count).clamp_min(1)
    pair_means = pair_means / pair_counts[:, None]
    centered_pair_means = pair_means - mean
    between_energy = (
        centered_pair_means.square().sum(1) * pair_counts.float()
    ).sum() / pair_counts.sum()
    total_energy = (residuals - mean).square().sum(1).mean()

    result = {
        "full_seen_pair_count": full_class_count,
        "probe_pair_count": class_count,
        "sample_count": len(labels),
        "chance_accuracy": 1.0 / class_count,
        "empirical_pair_probe_accuracy": centroid_probe(
            residuals, compact_labels, train_ids, test_ids, class_count
        ),
        "diagonal_gaussian_pair_probe_accuracy": centroid_probe(
            diag, compact_labels, train_ids, test_ids, class_count
        ),
        "isotropic_gaussian_pair_probe_accuracy": centroid_probe(
            iso, compact_labels, train_ids, test_ids, class_count
        ),
        "between_pair_energy_ratio": float(between_energy / total_energy.clamp_min(1e-12)),
        "residual_energy": float(total_energy),
        "isotropic_variance": float(iso_std.square()),
        "axis_max_to_mean_variance": float(axis_variance.max() / axis_variance.mean()),
        "axis_participation_rank": float(axis_participation_rank),
        "axis_participation_rank_fraction": float(
            axis_participation_rank / residuals.shape[1]
        ),
        "axis_entropy_rank": float(axis_entropy_rank),
        "axis_entropy_rank_fraction": float(axis_entropy_rank / residuals.shape[1]),
        "top_32_axis_energy_fraction": float(axis_variance.topk(32).values.sum() / axis_variance.sum()),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
