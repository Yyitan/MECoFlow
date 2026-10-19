"""Validation-only selection of reliability-gated generative evidence fusion.

The script deliberately does not access or evaluate the cached test split. It selects a single gate
family, threshold quantile, temperature and fusion strength from the mean
validation metrics across multiple independently generated classifier runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-seed-auc-gain", type=float, default=-0.001)
    parser.add_argument("--min-seed-hm-gain", type=float, default=-0.002)
    return parser.parse_args()


def row_zscore(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean(1, keepdim=True)) / value.std(1, keepdim=True).clamp_min(1e-6)


def location_scale(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust validation statistics, reusable without test-batch normalization."""
    center = value.median()
    q25, q75 = torch.quantile(value, torch.tensor([0.25, 0.75], device=value.device))
    return center, (q75 - q25).clamp_min(1e-6)


def standardize(value: torch.Tensor) -> torch.Tensor:
    center, scale = location_scale(value)
    return (value - center) / scale


def raw_reliability_features(
    base_z: torch.Tensor,
    generated_z: torch.Tensor,
    text_z: torch.Tensor,
) -> dict[str, torch.Tensor]:
    base_margin = base_z.topk(2, dim=1).values.diff(dim=1).neg().squeeze(1)
    generated_margin = generated_z.topk(2, dim=1).values.diff(dim=1).neg().squeeze(1)
    base_generated_corr = (base_z * generated_z).mean(1)
    text_generated_corr = (text_z * generated_z).mean(1)
    relative_margin = generated_margin - base_margin

    return {
        "generated_margin": generated_margin,
        "relative_margin": relative_margin,
        "base_agreement": base_generated_corr,
        "text_agreement": text_generated_corr,
    }


def fit_reliability_stats(
    raw: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {name: location_scale(value) for name, value in raw.items()}


def reliability_features(
    raw: dict[str, torch.Tensor],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    normalized = {
        name: (value - stats[name][0]) / stats[name][1]
        for name, value in raw.items()
    }
    return {
        "generated_margin": raw["generated_margin"],
        "relative_margin": raw["relative_margin"],
        "base_agreement": raw["base_agreement"],
        "margin_base_joint": 0.5
        * (normalized["generated_margin"] + normalized["base_agreement"]),
        "three_way_consensus": (
            normalized["generated_margin"]
            + normalized["base_agreement"]
            + normalized["text_agreement"]
        )
        / 3.0,
        "relative_consensus": (
            normalized["relative_margin"]
            + normalized["base_agreement"]
            + normalized["text_agreement"]
        )
        / 3.0,
    }


def gate_from_validation_feature(
    feature: torch.Tensor,
    threshold_quantile: float,
    temperature_scale: float,
) -> torch.Tensor:
    threshold = torch.quantile(feature, threshold_quantile)
    _center, iqr = location_scale(feature)
    temperature = (iqr * temperature_scale).clamp_min(1e-6)
    return torch.sigmoid((feature - threshold) / temperature)


def mean_metrics(stats: list[dict[str, float]]) -> tuple[float, float]:
    return (
        float(np.mean([item["AUC"] for item in stats])),
        float(np.mean([item["best_hm"] for item in stats])),
    )


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), device
    )

    base_logits = val["logits"].float()
    base_z = row_zscore(base_logits)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_z = row_zscore(val["com"].float() @ text.T)
    base_stats = evaluator(base_logits, val["attr"], val["obj"], val["pair"])

    runs: list[dict[str, object]] = []
    for path in args.classifier_logits:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        generated_z = row_zscore(saved["val"].float().to(device))
        raw_features = raw_reliability_features(base_z, generated_z, text_z)
        features = reliability_features(raw_features, fit_reliability_stats(raw_features))
        runs.append(
            {
                "path": path,
                "full": generated_z,
                "update": generated_z - text_z,
                "features": features,
            }
        )

    thresholds = (0.25, 0.5, 0.75)
    temperatures = (0.25, 0.5, 1.0)
    strengths = (0.1, 0.2, 0.5, 1.0, 2.0)
    candidates: list[dict[str, object]] = []
    best_safe: tuple[float, float, float, dict[str, object]] | None = None
    best_any: tuple[float, float, float, dict[str, object]] | None = None

    for feature_name in next(iter(runs))["features"]:
        for threshold_quantile in thresholds:
            for temperature_scale in temperatures:
                gates = [
                    gate_from_validation_feature(
                        run["features"][feature_name], threshold_quantile, temperature_scale
                    )
                    for run in runs
                ]
                for delta_name in ("full", "update"):
                    for strength in strengths:
                        stats = [
                            evaluator(
                                base_logits + strength * gate[:, None] * run[delta_name],
                                val["attr"],
                                val["obj"],
                                val["pair"],
                            )
                            for run, gate in zip(runs, gates)
                        ]
                        mean_auc, mean_hm = mean_metrics(stats)
                        auc_gains = [item["AUC"] - base_stats["AUC"] for item in stats]
                        hm_gains = [item["best_hm"] - base_stats["best_hm"] for item in stats]
                        normalized_score = (
                            (mean_auc - base_stats["AUC"]) / max(base_stats["AUC"], 1e-6)
                            + (mean_hm - base_stats["best_hm"])
                            / max(base_stats["best_hm"], 1e-6)
                        )
                        record = {
                            "feature": feature_name,
                            "threshold_quantile": threshold_quantile,
                            "temperature_scale": temperature_scale,
                            "delta": delta_name,
                            "strength": strength,
                            "mean_auc": mean_auc,
                            "mean_hm": mean_hm,
                            "mean_auc_gain": mean_auc - base_stats["AUC"],
                            "mean_hm_gain": mean_hm - base_stats["best_hm"],
                            "min_auc_gain": min(auc_gains),
                            "min_hm_gain": min(hm_gains),
                            "normalized_score": normalized_score,
                            "runs": stats,
                        }
                        candidates.append(record)
                        key = (normalized_score, mean_auc, mean_hm, record)
                        if best_any is None or key[:3] > best_any[:3]:
                            best_any = key
                        safe = (
                            min(auc_gains) >= args.min_seed_auc_gain
                            and min(hm_gains) >= args.min_seed_hm_gain
                            and mean_auc >= base_stats["AUC"]
                            and mean_hm >= base_stats["best_hm"]
                        )
                        if safe and (best_safe is None or key[:3] > best_safe[:3]):
                            best_safe = key

    selected = best_safe if best_safe is not None else best_any
    assert selected is not None
    selected_record = selected[3]
    candidates.sort(key=lambda item: item["normalized_score"], reverse=True)
    output = {
        "protocol": "validation_only_test_split_not_accessed_or_evaluated",
        "base_val": base_stats,
        "selection_safe": best_safe is not None,
        "selection": selected_record,
        "top_candidates": candidates[:25],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("protocol", "base_val", "selection_safe", "selection")}, indent=2))


if __name__ == "__main__":
    main()
