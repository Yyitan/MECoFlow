"""Validation-selected ANOVA interaction fusion for cached generative logits.

The generated classifier update is decomposed over the active attribute-object
candidate graph.  Only the non-additive interaction term is fused into the frozen
base logits, leaving primitive main effects anchored to CAMS.
"""

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
    parser.add_argument("--classifier-logits", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--evaluator-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=12)
    return parser.parse_args()


def zscore(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(1, keepdim=True)) / logits.std(
        1, keepdim=True
    ).clamp_min(1e-6)


def grouped_mean(
    values: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
    active: torch.Tensor,
) -> torch.Tensor:
    weighted = values * active[None]
    result = values.new_zeros(values.shape[0], group_count)
    result.index_add_(1, groups, weighted)
    counts = values.new_zeros(group_count)
    counts.index_add_(0, groups, active)
    return result / counts.clamp_min(1)[None]


def interaction_residual(
    scores: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    attr_count: int,
    obj_count: int,
    active: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    active_count = active.sum().clamp_min(1)
    grand = (scores * active[None]).sum(1, keepdim=True) / active_count
    centered = scores - grand
    attr_effect = scores.new_zeros(scores.shape[0], attr_count)
    obj_effect = scores.new_zeros(scores.shape[0], obj_count)
    for _ in range(iterations):
        attr_effect = grouped_mean(
            centered - obj_effect[:, pair_obj], pair_attr, attr_count, active
        )
        obj_effect = grouped_mean(
            centered - attr_effect[:, pair_attr], pair_obj, obj_count, active
        )
    residual = centered - attr_effect[:, pair_attr] - obj_effect[:, pair_obj]
    return residual * active[None]


def main() -> None:
    args = parse_args()
    evaluator_repo = Path(args.evaluator_repo).resolve()
    sys.path.insert(0, str(evaluator_repo))
    from research.fast_feature_evaluation import (  # pylint: disable=import-outside-toplevel
        FastClosedWorldEvaluator,
        FastOpenWorldEvaluator,
    )
    from dataset import CompositionDataset  # pylint: disable=import-outside-toplevel

    device = torch.device("cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    generated = torch.load(
        args.classifier_logits, map_location="cpu", weights_only=False
    )
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    active = torch.tensor(
        meta.get("active", [True] * len(meta["pairs"])), dtype=torch.float32
    )

    if meta.get("open_world", False):
        val_eval = FastOpenWorldEvaluator(cache, device)
        test_eval = FastOpenWorldEvaluator(cache, device)
    else:
        val_eval = FastClosedWorldEvaluator(
            CompositionDataset(args.dataset_path, phase="val"), device
        )
        test_eval = FastClosedWorldEvaluator(
            CompositionDataset(args.dataset_path, phase="test"), device
        )

    prompts = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    strengths = (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05,
                 0.1, 0.2, 0.5, 1.0, 2.0)
    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split in ("val", "test"):
        base = cache[split]["logits"].float()
        text = zscore(cache[split]["com"].float() @ prompts.T)
        full = zscore(generated[split].float()) * active[None]
        update = (full - text) * active[None]
        splits[split] = {
            "base": base,
            "full_interaction": interaction_residual(
                full, pair_attr, pair_obj, len(attrs), len(objs), active,
                args.iterations,
            ),
            "update_interaction": interaction_residual(
                update, pair_attr, pair_obj, len(attrs), len(objs), active,
                args.iterations,
            ),
        }

    val_truth = cache["val"]
    trials: dict[str, dict[str, float]] = {}
    best = None
    for name in ("full_interaction", "update_interaction"):
        for strength in strengths:
            logits = splits["val"]["base"] + strength * splits["val"][name]
            stats = val_eval(
                logits, val_truth["attr"], val_truth["obj"], val_truth["pair"]
            )
            key = f"{name}_{strength:.2f}"
            trials[key] = stats
            candidate = (stats["AUC"], stats["best_hm"], name, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    _, _, name, strength = best
    test_truth = cache["test"]
    test_logits = splits["test"]["base"] + strength * splits["test"][name]
    result = {
        "base_val": val_eval(
            splits["val"]["base"], val_truth["attr"], val_truth["obj"],
            val_truth["pair"],
        ),
        "base_test": test_eval(
            splits["test"]["base"], test_truth["attr"], test_truth["obj"],
            test_truth["pair"],
        ),
        "selection": {
            "variant": name,
            "strength": strength,
            "val_auc": best[0],
            "val_hm": best[1],
        },
        "fused_test": test_eval(
            test_logits, test_truth["attr"], test_truth["obj"], test_truth["pair"]
        ),
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "trials"}, indent=2))


if __name__ == "__main__":
    main()
