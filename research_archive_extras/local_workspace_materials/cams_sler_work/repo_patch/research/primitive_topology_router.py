"""Pair-held-out primitive-topology router for generative CZSL evidence.

Validation labels supervise whether a strong generated update improves the correct
margin, but every prediction is out-of-fold for the complete ground-truth pair.
The topology model can only memorize primitive-node reliability, never a held-out
composition identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", required=True)
    parser.add_argument("--delta", choices=("full", "update"), default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(1, keepdim=True)) / logits.std(
        1, keepdim=True
    ).clamp_min(1e-6)


def correct_margin(
    logits: torch.Tensor, truth: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    rows = torch.arange(len(logits))
    truth_score = logits[rows, truth]
    competitor = logits.masked_fill(
        ~candidate_mask[None], torch.finfo(logits.dtype).min
    ).clone()
    competitor[rows, truth] = torch.finfo(logits.dtype).min
    return truth_score - competitor.max(1).values


def binary_auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    score, positive = score.detach().float().cpu(), positive.detach().bool().cpu()
    num_positive = int(positive.sum())
    num_negative = len(positive) - num_positive
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    return float(
        (ranks[positive].sum().item() - num_positive * (num_positive + 1) / 2)
        / (num_positive * num_negative)
    )


def numeric_features(
    base: torch.Tensor, generated: torch.Tensor, closed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    closed_ids = torch.where(closed)[0]
    base_closed, generated_closed = base[:, closed], generated[:, closed]
    base_values, base_local = base_closed.topk(5, dim=1)
    gen_values, gen_local = generated_closed.topk(5, dim=1)
    base_top = closed_ids[base_local[:, 0]]
    gen_top = closed_ids[gen_local[:, 0]]
    base_probability = torch.softmax(base_closed, dim=1)
    gen_probability = torch.softmax(generated_closed, dim=1)
    mixture = 0.5 * (base_probability + gen_probability)
    log_count = np.log(base_closed.shape[1])
    base_entropy = -(
        base_probability * base_probability.clamp_min(1e-12).log()
    ).sum(1) / log_count
    gen_entropy = -(
        gen_probability * gen_probability.clamp_min(1e-12).log()
    ).sum(1) / log_count
    js = 0.5 * (
        (
            base_probability
            * (base_probability.clamp_min(1e-12) / mixture.clamp_min(1e-12)).log()
        ).sum(1)
        + (
            gen_probability
            * (gen_probability.clamp_min(1e-12) / mixture.clamp_min(1e-12)).log()
        ).sum(1)
    )
    overlap = (
        base_local[:, :, None] == gen_local[:, None, :]
    ).any(2).float().mean(1)
    correlation = (zscore(base_closed) * zscore(generated_closed)).mean(1)
    columns = [
        base_values[:, 0] - base_values[:, 1],
        gen_values[:, 0] - gen_values[:, 1],
        (gen_values[:, 0] - gen_values[:, 1])
        - (base_values[:, 0] - base_values[:, 1]),
        (base_top == gen_top).float(),
        overlap,
        correlation,
        base_entropy,
        gen_entropy,
        js,
        (generated_closed - base_closed).mean(1),
        (generated_closed - base_closed).std(1),
        (generated_closed - base_closed).max(1).values,
    ]
    names = [
        "base_margin", "generated_margin", "relative_margin", "top1_agreement",
        "top5_overlap", "score_correlation", "base_entropy", "generated_entropy",
        "js_divergence", "delta_mean", "delta_std", "delta_max",
    ]
    return torch.stack(columns, 1), base_top, gen_top, names


class PrimitiveTopologyRouter(nn.Module):
    def __init__(
        self, numeric_dim: int, attr_count: int, obj_count: int, use_topology: bool
    ) -> None:
        super().__init__()
        self.numeric = nn.Linear(numeric_dim, 1)
        self.use_topology = use_topology
        if use_topology:
            self.base_attr = nn.Embedding(attr_count, 1)
            self.base_obj = nn.Embedding(obj_count, 1)
            self.gen_attr = nn.Embedding(attr_count, 1)
            self.gen_obj = nn.Embedding(obj_count, 1)
            for embedding in (
                self.base_attr, self.base_obj, self.gen_attr, self.gen_obj
            ):
                nn.init.zeros_(embedding.weight)

    def forward(
        self,
        numeric: torch.Tensor,
        base_attr: torch.Tensor,
        base_obj: torch.Tensor,
        gen_attr: torch.Tensor,
        gen_obj: torch.Tensor,
    ) -> torch.Tensor:
        output = self.numeric(numeric).squeeze(1)
        if self.use_topology:
            output = output + self.base_attr(base_attr).squeeze(1)
            output = output + self.base_obj(base_obj).squeeze(1)
            output = output + self.gen_attr(gen_attr).squeeze(1)
            output = output + self.gen_obj(gen_obj).squeeze(1)
        return output


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    generated_saved = torch.load(
        args.classifier_logits, map_location="cpu", weights_only=True
    )
    val = cache["val"]
    base = val["logits"].float()
    base_z = zscore(base)
    generated = zscore(generated_saved["val"].float())
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_z = zscore(val["com"].float() @ text.T)
    delta = generated if args.delta == "full" else generated - text_z

    dataset = CompositionDataset(args.dataset_path, phase="val")
    evaluator = FastClosedWorldEvaluator(dataset, torch.device("cpu"))
    pair_attr = torch.tensor([dataset.attr2idx[a] for a, _ in dataset.pairs])
    pair_obj = torch.tensor([dataset.obj2idx[o] for _, o in dataset.pairs])
    numeric, base_top, gen_top, feature_names = numeric_features(
        base_z, generated, evaluator.closed
    )
    base_attr, base_obj = pair_attr[base_top], pair_obj[base_top]
    gen_attr, gen_obj = pair_attr[gen_top], pair_obj[gen_top]

    base_margin = correct_margin(base, val["pair"], evaluator.closed)
    strong_margin = correct_margin(base + 4.0 * delta, val["pair"], evaluator.closed)
    target = (strong_margin > base_margin).float()

    unique_pairs = torch.unique(val["pair"])
    generator = torch.Generator().manual_seed(args.seed)
    unique_pairs = unique_pairs[torch.randperm(len(unique_pairs), generator=generator)]
    pair_fold = torch.full((len(dataset.pairs),), -1, dtype=torch.long)
    pair_fold[unique_pairs] = torch.arange(len(unique_pairs)) % args.folds
    fold_ids = pair_fold[val["pair"]]

    model_outputs: dict[str, object] = {}
    for model_name, use_topology in (("numeric", False), ("topology", True)):
        oof_logits = torch.empty(len(numeric))
        fold_metrics = []
        for fold in range(args.folds):
            train_mask, heldout_mask = fold_ids != fold, fold_ids == fold
            center = numeric[train_mask].mean(0)
            scale = numeric[train_mask].std(0).clamp_min(1e-6)
            train_numeric = (numeric[train_mask] - center) / scale
            heldout_numeric = (numeric[heldout_mask] - center) / scale
            train_target = target[train_mask]
            model = PrimitiveTopologyRouter(
                numeric.shape[1], len(dataset.attrs), len(dataset.objs), use_topology
            )
            positive = train_target.sum().clamp_min(1)
            pos_weight = (len(train_target) - positive) / positive
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
            for _ in range(args.epochs):
                prediction = model(
                    train_numeric,
                    base_attr[train_mask], base_obj[train_mask],
                    gen_attr[train_mask], gen_obj[train_mask],
                )
                loss = F.binary_cross_entropy_with_logits(
                    prediction, train_target, pos_weight=pos_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                prediction = model(
                    heldout_numeric,
                    base_attr[heldout_mask], base_obj[heldout_mask],
                    gen_attr[heldout_mask], gen_obj[heldout_mask],
                )
                oof_logits[heldout_mask] = prediction
                fold_metrics.append({
                    "fold": fold,
                    "samples": int(heldout_mask.sum()),
                    "pairs": int(torch.unique(val["pair"][heldout_mask]).numel()),
                    "target_rate": float(target[heldout_mask].mean()),
                    "auc": binary_auc(prediction, target[heldout_mask] > 0),
                })
        probability = torch.sigmoid(oof_logits)
        trials = []
        best = None
        gate_candidates = [
            ("soft", probability),
            ("hard_03", (probability >= 0.3).float()),
            ("hard_04", (probability >= 0.4).float()),
            ("hard_05", (probability >= 0.5).float()),
            ("hard_06", (probability >= 0.6).float()),
            ("hard_07", (probability >= 0.7).float()),
            ("hard_08", (probability >= 0.8).float()),
            ("hard_09", (probability >= 0.9).float()),
            ("hard_095", (probability >= 0.95).float()),
        ]
        for fraction in (0.05, 0.10, 0.20, 0.30):
            threshold = torch.quantile(probability, 1.0 - fraction)
            gate_candidates.append(
                (f"top_{int(100 * fraction):02d}pct", (probability >= threshold).float())
            )
        for gate_name, gate in gate_candidates:
            for strength in (0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
                stats = evaluator(
                    base + strength * gate[:, None] * delta,
                    val["attr"], val["obj"], val["pair"],
                )
                record = {
                    "gate": gate_name,
                    "strength": strength,
                    "AUC": stats["AUC"],
                    "best_hm": stats["best_hm"],
                }
                trials.append(record)
                candidate = (stats["AUC"], stats["best_hm"], record)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        assert best is not None
        model_outputs[model_name] = {
            "oof_auc": binary_auc(oof_logits, target > 0),
            "folds": fold_metrics,
            "best_oof_fusion": best[2],
            "trials": trials,
        }

    base_stats = evaluator(base, val["attr"], val["obj"], val["pair"])
    fixed_stats = evaluator(
        base + 0.2 * delta, val["attr"], val["obj"], val["pair"]
    )
    output = {
        "protocol": "validation_only_true_pair_held_out_topology_router",
        "feature_names": feature_names,
        "target": "correct_margin_improves_at_generated_strength_4",
        "target_rate": float(target.mean()),
        "base_val": base_stats,
        "fixed_strength_0.2": fixed_stats,
        "models": model_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary = dict(output)
    summary["models"] = {
        name: {key: value for key, value in result.items() if key != "trials"}
        for name, result in model_outputs.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
