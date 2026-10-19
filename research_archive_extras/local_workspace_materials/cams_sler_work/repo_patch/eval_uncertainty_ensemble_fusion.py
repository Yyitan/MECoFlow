"""Select or apply candidate-wise uncertainty-aware generative fusion.

Independent generated classifiers define a mean and epistemic standard deviation
for every image/composition candidate.  Validation selects whether to use the mean,
a lower confidence bound, or variance shrinkage.  Apply mode consumes the frozen
selection and evaluates the held-out test split without any further search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_reliability_gated_fusion import row_zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "apply"), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ensemble_statistics(
    saved: list[dict[str, torch.Tensor]], split: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    members = torch.stack(
        [row_zscore(item[split].float().to(device)) for item in saved]
    )
    return members.mean(0), members.std(0, unbiased=True).clamp_min(1e-6)


def risk_adjusted(mean: torch.Tensor, std: torch.Tensor, kind: str, risk: float) -> torch.Tensor:
    if kind == "mean":
        return mean
    if kind == "lcb":
        return mean - risk * std
    if kind == "shrink":
        return mean / (1.0 + risk * std)
    if kind == "soft_lcb":
        return mean - risk * std * torch.sigmoid(mean)
    raise ValueError(kind)


def evaluate(
    evaluator: object,
    logits: torch.Tensor,
    split: dict[str, torch.Tensor],
) -> dict[str, float]:
    return evaluator(logits, split["attr"], split["obj"], split["pair"])


def uncertainty_summary(std: torch.Tensor) -> dict[str, float]:
    """Summarize very large score tensors without CUDA quantile size limits."""
    flat = std.reshape(-1)
    stride = max((flat.numel() + 999_999) // 1_000_000, 1)
    sample = flat[::stride].float().cpu()
    return {
        "mean": float(std.mean()),
        "std": float(std.std()),
        "q50": float(torch.quantile(sample, 0.5)),
        "q90": float(torch.quantile(sample, 0.9)),
        "quantile_sample_size": len(sample),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.classifier_logits
    ]
    split_name = "val" if args.mode == "select" else "test"
    phase = "val" if args.mode == "select" else "test"
    split = {key: value.to(device) for key, value in cache[split_name].items()}
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase=phase), device
    )
    base = split["logits"].float()
    base_stats = evaluate(evaluator, base, split)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_z = row_zscore(split["com"].float() @ text.T)
    mean, std = ensemble_statistics(saved, split_name, device)

    if args.mode == "select":
        risks = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
        strengths = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
        candidates: list[dict[str, object]] = []
        best: tuple[float, float, float, dict[str, object]] | None = None
        for kind in ("mean", "lcb", "shrink", "soft_lcb"):
            for risk in risks:
                if kind == "mean" and risk != 0.0:
                    continue
                adjusted = risk_adjusted(mean, std, kind, risk)
                for delta_name, delta in (
                    ("full", adjusted),
                    ("update", adjusted - text_z),
                ):
                    for strength in strengths:
                        stats = evaluate(evaluator, base + strength * delta, split)
                        auc_gain = stats["AUC"] - base_stats["AUC"]
                        hm_gain = stats["best_hm"] - base_stats["best_hm"]
                        score = (
                            auc_gain / max(base_stats["AUC"], 1e-6)
                            + hm_gain / max(base_stats["best_hm"], 1e-6)
                        )
                        record = {
                            "kind": kind,
                            "risk": risk,
                            "delta": delta_name,
                            "strength": strength,
                            "AUC": stats["AUC"],
                            "best_hm": stats["best_hm"],
                            "auc_gain": auc_gain,
                            "hm_gain": hm_gain,
                            "normalized_score": score,
                            "stats": stats,
                        }
                        candidates.append(record)
                        safe = auc_gain >= 0.0 and hm_gain >= 0.0
                        key = (score, stats["AUC"], stats["best_hm"], record)
                        if safe and (best is None or key[:3] > best[:3]):
                            best = key
        if best is None:
            raise RuntimeError("No candidate improved both validation AUC and HM")
        candidates.sort(key=lambda item: item["normalized_score"], reverse=True)
        output = {
            "protocol": "validation_only_test_split_not_accessed_or_evaluated",
            "base_val": base_stats,
            "selection": best[3],
            "top_candidates": candidates[:30],
            "ensemble_uncertainty": uncertainty_summary(std),
        }
    else:
        if not args.selection:
            raise ValueError("--selection is required in apply mode")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selection"]
        adjusted = risk_adjusted(mean, std, selection["kind"], selection["risk"])
        delta = adjusted if selection["delta"] == "full" else adjusted - text_z
        fused_stats = evaluate(
            evaluator, base + selection["strength"] * delta, split
        )
        output = {
            "protocol": "test_application_of_frozen_validation_selection",
            "selection": selection,
            "base_test": base_stats,
            "fused_test": fused_stats,
            "gain": {
                "AUC": fused_stats["AUC"] - base_stats["AUC"],
                "best_hm": fused_stats["best_hm"] - base_stats["best_hm"],
            },
            "ensemble_uncertainty": uncertainty_summary(std),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in output if key != "top_candidates"}, indent=2))


if __name__ == "__main__":
    main()
