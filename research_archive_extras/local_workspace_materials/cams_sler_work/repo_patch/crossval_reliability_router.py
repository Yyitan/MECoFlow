"""Pair-held-out diagnostic for learning when generated evidence is useful.

This script is intentionally validation-only. It derives oracle-use labels from the
validation ground truth, but every prediction is out-of-fold for the complete target
composition. The result tests whether a richer label-free reliability representation
is learnable before investing in pseudo-unseen episodic training on train pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval_reliability_gated_fusion import row_zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--delta", choices=("full", "update"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def correct_margin(
    logits: torch.Tensor, truth: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    rows = torch.arange(len(logits), device=logits.device)
    truth_score = logits[rows, truth]
    competitors = logits.masked_fill(~candidate_mask[None], torch.finfo(logits.dtype).min)
    competitors[rows, truth] = torch.finfo(logits.dtype).min
    return truth_score - competitors.max(1).values


def binary_auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    score, positive = score.float().cpu(), positive.bool().cpu()
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    return float(
        (ranks[positive].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def pair_maps(dataset: object, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([dataset.attr2idx[a] for a, _ in dataset.pairs], device=device),
        torch.tensor([dataset.obj2idx[o] for _, o in dataset.pairs], device=device),
    )


def extract_features(
    base_z: torch.Tensor,
    generated_z: torch.Tensor,
    generated_std: torch.Tensor,
    closed: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
) -> tuple[torch.Tensor, list[str]]:
    base = base_z[:, closed]
    generated = generated_z[:, closed]
    uncertainty = generated_std[:, closed]
    closed_ids = torch.where(closed)[0]
    topk = min(5, base.shape[1])
    base_values, base_local = base.topk(topk, dim=1)
    generated_values, generated_local = generated.topk(topk, dim=1)
    base_top = closed_ids[base_local[:, 0]]
    generated_top = closed_ids[generated_local[:, 0]]
    rows = torch.arange(len(base), device=base.device)
    base_margin = base_values[:, 0] - base_values[:, 1]
    generated_margin = generated_values[:, 0] - generated_values[:, 1]
    overlap = (
        base_local[:, :, None] == generated_local[:, None, :]
    ).any(2).float().mean(1)
    correlation = (row_zscore(base) * row_zscore(generated)).mean(1)
    base_prob = torch.softmax(base, dim=1)
    generated_prob = torch.softmax(generated, dim=1)
    log_classes = math.log(base.shape[1])
    base_entropy = -(base_prob * base_prob.clamp_min(1e-12).log()).sum(1) / log_classes
    generated_entropy = -(
        generated_prob * generated_prob.clamp_min(1e-12).log()
    ).sum(1) / log_classes
    mixture = 0.5 * (base_prob + generated_prob)
    js = 0.5 * (
        (base_prob * (base_prob.clamp_min(1e-12) / mixture.clamp_min(1e-12)).log()).sum(1)
        + (
            generated_prob
            * (generated_prob.clamp_min(1e-12) / mixture.clamp_min(1e-12)).log()
        ).sum(1)
    )
    delta = generated - base
    std_at_base = generated_std[rows, base_top]
    std_at_generated = generated_std[rows, generated_top]
    uncertainty_top5 = uncertainty.gather(1, generated_local).mean(1)
    feature_columns = [
        base_margin,
        generated_margin,
        generated_margin - base_margin,
        (base_top == generated_top).float(),
        overlap,
        correlation,
        base_entropy,
        generated_entropy,
        js,
        generated[rows, base_local[:, 0]] - generated_values[:, 0],
        base[rows, generated_local[:, 0]] - base_values[:, 0],
        delta.mean(1),
        delta.std(1),
        delta.max(1).values,
        std_at_base,
        std_at_generated,
        uncertainty_top5,
        (pair_attr[base_top] == pair_attr[generated_top]).float(),
        (pair_obj[base_top] == pair_obj[generated_top]).float(),
    ]
    names = [
        "base_margin", "generated_margin", "relative_margin", "top1_agreement",
        "top5_overlap", "score_correlation", "base_entropy", "generated_entropy",
        "js_divergence", "generated_score_at_base_top_gap", "base_score_at_generated_top_gap",
        "delta_mean", "delta_std", "delta_max", "uncertainty_at_base_top",
        "uncertainty_at_generated_top", "uncertainty_generated_top5",
        "top_attr_agreement", "top_obj_agreement",
    ]
    return torch.stack(feature_columns, dim=1), names


class Router(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    saved = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.classifier_logits
    ]
    members = torch.stack(
        [row_zscore(item["val"].float().to(device)) for item in saved]
    )
    generated, generated_std = members.mean(0), members.std(0, unbiased=True)
    base = val["logits"].float()
    base_z = row_zscore(base)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    text_z = row_zscore(val["com"].float() @ text.T)
    delta = generated if args.delta == "full" else generated - text_z

    dataset = CompositionDataset(args.dataset_path, phase="val")
    evaluator = FastClosedWorldEvaluator(dataset, device)
    pair_attr, pair_obj = pair_maps(dataset, device)
    features, feature_names = extract_features(
        base_z, generated, generated_std, evaluator.closed, pair_attr, pair_obj
    )
    base_margin = correct_margin(base, val["pair"], evaluator.closed)
    strengths = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
    best_margin = torch.full_like(base_margin, -torch.inf)
    best_strength = torch.zeros_like(base_margin)
    for strength in strengths:
        margin = correct_margin(base + strength * delta, val["pair"], evaluator.closed)
        better = margin > best_margin
        best_margin[better] = margin[better]
        best_strength[better] = strength
    target = (best_strength > 0).float()

    unique_pairs = torch.unique(val["pair"]).cpu()
    generator = torch.Generator().manual_seed(args.seed)
    unique_pairs = unique_pairs[torch.randperm(len(unique_pairs), generator=generator)]
    pair_fold = {int(pair): index % args.folds for index, pair in enumerate(unique_pairs)}
    fold_ids = torch.tensor([pair_fold[int(pair)] for pair in val["pair"].cpu()], device=device)
    oof_logits = torch.empty(len(features), device=device)
    fold_results: list[dict[str, float]] = []
    for fold in range(args.folds):
        train_mask, heldout_mask = fold_ids != fold, fold_ids == fold
        center = features[train_mask].mean(0)
        scale = features[train_mask].std(0).clamp_min(1e-6)
        train_x = (features[train_mask] - center) / scale
        heldout_x = (features[heldout_mask] - center) / scale
        train_y = target[train_mask]
        model = Router(features.shape[1]).to(device)
        positive = train_y.sum().clamp_min(1)
        pos_weight = (len(train_y) - positive) / positive
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        for _ in range(args.epochs):
            logits = model(train_x)
            loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = model(heldout_x)
            oof_logits[heldout_mask] = prediction
            fold_results.append(
                {
                    "fold": fold,
                    "samples": int(heldout_mask.sum()),
                    "pairs": int(torch.unique(val["pair"][heldout_mask]).numel()),
                    "positive_rate": float(target[heldout_mask].mean()),
                    "auc": binary_auc(prediction, target[heldout_mask] > 0),
                }
            )

    probability = torch.sigmoid(oof_logits)
    base_stats = evaluator(base, val["attr"], val["obj"], val["pair"])
    trials: list[dict[str, object]] = []
    best = None
    for gate_kind, gate in (
        ("soft", probability),
        ("hard_03", (probability >= 0.3).float()),
        ("hard_05", (probability >= 0.5).float()),
        ("hard_07", (probability >= 0.7).float()),
    ):
        for strength in (0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
            stats = evaluator(
                base + strength * gate[:, None] * delta,
                val["attr"], val["obj"], val["pair"],
            )
            record = {
                "gate": gate_kind,
                "strength": strength,
                "AUC": stats["AUC"],
                "best_hm": stats["best_hm"],
                "auc_gain": stats["AUC"] - base_stats["AUC"],
                "hm_gain": stats["best_hm"] - base_stats["best_hm"],
                "stats": stats,
            }
            trials.append(record)
            candidate = (stats["AUC"] + stats["best_hm"], stats["AUC"], stats["best_hm"], record)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    assert best is not None
    output = {
        "protocol": "validation_only_pair_held_out_cross_validation_diagnostic",
        "feature_names": feature_names,
        "target_positive_rate": float(target.mean()),
        "oof_auc": binary_auc(oof_logits, target > 0),
        "folds": fold_results,
        "base_val": base_stats,
        "best_oof_fusion": best[3],
        "trials": trials,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in output if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
