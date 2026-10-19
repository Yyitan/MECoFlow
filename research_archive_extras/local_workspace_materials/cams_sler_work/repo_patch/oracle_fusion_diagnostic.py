"""Validation-only oracle analysis for generated-classifier complementarity.

This is a diagnostic, not a reportable method: it uses validation labels to choose
the best fusion strength independently for every image.  Its purpose is to decide
whether better label-free routing has meaningful headroom or whether the generated
evidence itself must be improved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_reliability_gated_fusion import (
    fit_reliability_stats,
    raw_reliability_features,
    reliability_features,
    row_zscore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--fixed-delta", choices=("full", "update"), required=True)
    parser.add_argument("--fixed-strength", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def binary_auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    """Mann-Whitney AUROC with deterministic ordinal ranks (diagnostic only)."""
    score = score.detach().float().cpu()
    positive = positive.detach().bool().cpu()
    num_positive = int(positive.sum())
    num_negative = len(positive) - num_positive
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    rank_sum = ranks[positive].sum().item()
    return float(
        (rank_sum - num_positive * (num_positive + 1) / 2)
        / (num_positive * num_negative)
    )


def correct_margin(
    logits: torch.Tensor, truth: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    index = torch.arange(len(logits), device=logits.device)
    truth_score = logits[index, truth]
    competitor = logits.masked_fill(~candidate_mask[None], torch.finfo(logits.dtype).min)
    competitor[index, truth] = torch.finfo(logits.dtype).min
    return truth_score - competitor.max(1).values


def mean_stats(items: list[dict[str, float]]) -> dict[str, float]:
    keys = ("AUC", "best_hm", "best_seen", "best_unseen")
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    dataset = CompositionDataset(args.dataset_path, phase="val")
    evaluator = FastClosedWorldEvaluator(dataset, device)
    base = val["logits"].float()
    base_z = row_zscore(base)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_z = row_zscore(val["com"].float() @ text.T)
    truth = val["pair"]
    base_stats = evaluator(base, val["attr"], val["obj"], truth)
    base_margin = correct_margin(base, truth, evaluator.closed)

    strengths = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
    run_outputs: list[dict[str, object]] = []
    fixed_stats_all: list[dict[str, float]] = []
    oracle_stats_all: list[dict[str, float]] = []
    for path in args.classifier_logits:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        generated = row_zscore(saved["val"].float().to(device))
        delta = generated if args.fixed_delta == "full" else generated - text_z
        fixed = base + args.fixed_strength * delta
        fixed_stats = evaluator(fixed, val["attr"], val["obj"], truth)

        best_margin = torch.full_like(base_margin, -torch.inf)
        best_strength = torch.zeros_like(base_margin)
        oracle_logits = base.clone()
        for strength in strengths:
            fused = base + strength * delta
            margin = correct_margin(fused, truth, evaluator.closed)
            better = margin > best_margin
            best_margin[better] = margin[better]
            best_strength[better] = strength
            oracle_logits[better] = fused[better]
        oracle_stats = evaluator(oracle_logits, val["attr"], val["obj"], truth)

        raw_features = raw_reliability_features(base_z, generated, text_z)
        features = reliability_features(raw_features, fit_reliability_stats(raw_features))
        oracle_uses_generation = best_strength > 0
        oracle_changes_decision = best_strength != args.fixed_strength
        predictiveness = {
            name: {
                "auc_for_oracle_use": binary_auc(value, oracle_uses_generation),
                "auc_for_fixed_margin_improvement": binary_auc(
                    value, correct_margin(fixed, truth, evaluator.closed) > base_margin
                ),
            }
            for name, value in features.items()
        }
        run_outputs.append(
            {
                "path": path,
                "fixed": fixed_stats,
                "oracle": oracle_stats,
                "oracle_use_rate": float(oracle_uses_generation.float().mean()),
                "oracle_changes_from_fixed_rate": float(
                    oracle_changes_decision.float().mean()
                ),
                "oracle_strength_histogram": {
                    f"{strength:g}": float((best_strength == strength).float().mean())
                    for strength in strengths
                },
                "mean_margin_gain_fixed": float(
                    (correct_margin(fixed, truth, evaluator.closed) - base_margin).mean()
                ),
                "mean_margin_gain_oracle": float((best_margin - base_margin).mean()),
                "feature_predictiveness": predictiveness,
            }
        )
        fixed_stats_all.append(fixed_stats)
        oracle_stats_all.append(oracle_stats)

    output = {
        "protocol": "validation_only_label_using_oracle_diagnostic",
        "base": base_stats,
        "fixed_configuration": {
            "delta": args.fixed_delta,
            "strength": args.fixed_strength,
        },
        "fixed_mean": mean_stats(fixed_stats_all),
        "oracle_mean": mean_stats(oracle_stats_all),
        "oracle_gain_over_fixed": {
            key: mean_stats(oracle_stats_all)[key] - mean_stats(fixed_stats_all)[key]
            for key in ("AUC", "best_hm")
        },
        "runs": run_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: output[key] for key in (
                "protocol", "base", "fixed_configuration", "fixed_mean",
                "oracle_mean", "oracle_gain_over_fixed",
            )},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
