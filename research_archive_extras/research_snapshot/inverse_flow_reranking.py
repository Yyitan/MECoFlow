"""Retrieval-guided inverse-flow energy for compositional reranking."""

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

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids  # noqa: E402
from research.piflow_pilot import PrimitiveConditionalFlow  # noqa: E402
from research.warmstart_flow_classifier import pair_maps, residual_bank  # noqa: E402


def shortlist(logits: torch.Tensor, seen: torch.Tensor, topk: int) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min
    k_seen = min(topk, int(seen.sum()))
    k_unseen = min(topk, int((~seen).sum()))
    seen_ids = logits.masked_fill(~seen[None], neg).topk(k_seen, dim=1).indices
    unseen_ids = logits.masked_fill(seen[None], neg).topk(k_unseen, dim=1).indices
    return torch.cat([seen_ids, unseen_ids], dim=1)


def group_zscore(value: torch.Tensor, topk: int) -> torch.Tensor:
    groups = []
    for chunk in (value[:, :topk], value[:, topk:]):
        groups.append((chunk - chunk.mean(1, keepdim=True)) / chunk.std(1, keepdim=True).clamp_min(1e-4))
    return torch.cat(groups, dim=1)


@torch.no_grad()
def inverse_energy(
    features: torch.Tensor,
    candidates: torch.Tensor,
    model: PrimitiveConditionalFlow,
    additive: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    attr_prompt: torch.Tensor,
    obj_prompt: torch.Tensor,
    pair_prompt: torch.Tensor,
    residual_var: torch.Tensor,
    residual_norm: torch.Tensor,
    steps: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    rows = torch.arange(len(features), device=features.device)[:, None].expand_as(candidates).reshape(-1)
    labels = candidates.reshape(-1)
    outputs = {name: [] for name in ("cosine", "l2", "mahalanobis", "norm")}
    dt = 1.0 / steps
    for offset in range(0, len(labels), batch_size):
        index = slice(offset, offset + batch_size)
        row, label = rows[index], labels[index]
        target = features[row]
        current = target
        # Backward Euler through the learned conditional transport.  A correct
        # composition should map a real image back near that composition's
        # primitive residual source distribution.
        for step in range(steps):
            time = torch.full(
                (len(label),), 1.0 - step * dt, device=features.device
            )
            velocity = model(
                current,
                time,
                attr_prompt[pair_attr[label]],
                obj_prompt[pair_obj[label]],
                pair_prompt[label],
            )
            current = current - dt * velocity
        prior = additive[label]
        normalized = F.normalize(current, dim=-1)
        difference = current - prior
        outputs["cosine"].append((normalized * prior).sum(-1))
        outputs["l2"].append(-difference.square().sum(-1))
        outputs["mahalanobis"].append(-(difference.square() / residual_var).mean(-1))
        outputs["norm"].append(-(difference.norm(dim=-1) - residual_norm).abs())
    shape = candidates.shape
    return {name: torch.cat(parts).reshape(shape) for name, parts in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--additive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()

    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    dim = cache["train"]["com"].shape[1]
    model = PrimitiveConditionalFlow(dim, args.hidden_dim, args.cond_dim).to(device)
    model.load_state_dict(torch.load(args.flow, map_location="cpu", weights_only=True))
    model.eval()
    additive = F.normalize(
        torch.load(args.additive, map_location="cpu", weights_only=True).float(), dim=-1
    ).to(device)
    pair_attr, pair_obj = pair_maps(cache)
    pair_attr, pair_obj = pair_attr.to(device), pair_obj.to(device)
    attr_prompt = cache["prompts"]["attr"].float().to(device)
    obj_prompt = cache["prompts"]["obj"].float().to(device)
    pair_prompt = cache["prompts"]["pair"].float().to(device)
    _centroids, counts = real_centroids(cache)
    seen = (counts > 0).to(device)
    residuals = residual_bank(cache, additive.cpu()).to(device)
    residual_var = residuals.var(0, unbiased=False).clamp_min(1e-4)
    residual_var = residual_var / residual_var.mean()
    residual_norm = residuals.norm(dim=-1).mean()

    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_candidates = shortlist(val["logits"].float(), seen, args.topk)
    test_candidates = shortlist(test["logits"].float(), seen, args.topk)
    val_energy = inverse_energy(
        val["com"].float(), val_candidates, model, additive, pair_attr, pair_obj,
        attr_prompt, obj_prompt, pair_prompt, residual_var, residual_norm,
        args.steps, args.batch_size,
    )
    test_energy = inverse_energy(
        test["com"].float(), test_candidates, model, additive, pair_attr, pair_obj,
        attr_prompt, obj_prompt, pair_prompt, residual_var, residual_norm,
        args.steps, args.batch_size,
    )
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])

    trials: dict[str, object] = {}
    best = None
    strengths = (-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5)
    for energy_name, value in val_energy.items():
        value = group_zscore(value, args.topk)
        for strength in strengths:
            logits = val["logits"].float().clone()
            logits.scatter_add_(1, val_candidates, strength * value)
            stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
            name = f"{energy_name}_{strength:g}"
            trials[name] = stats
            candidate = (stats["AUC"], stats["best_hm"], energy_name, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    gain = best[0] - base_val["AUC"]
    enabled = gain >= args.min_val_auc_gain
    applied_strength = best[3] if enabled else 0.0
    test_value = group_zscore(test_energy[best[2]], args.topk)
    raw_logits = test["logits"].float().clone()
    raw_logits.scatter_add_(1, test_candidates, best[3] * test_value)
    selected_logits = test["logits"].float().clone()
    selected_logits.scatter_add_(1, test_candidates, applied_strength * test_value)
    output = {
        "base_val": base_val,
        "base_test": base_test,
        "selection": {
            "energy": best[2], "strength": applied_strength,
            "raw_strength": best[3], "topk_per_partition": args.topk,
            "steps": args.steps, "val_auc": best[0], "val_hm": best[1],
            "val_auc_gain": gain, "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": enabled,
        },
        "raw_test": test_eval(raw_logits, test["attr"], test["obj"], test["pair"]),
        "selected_test": test_eval(selected_logits, test["attr"], test["obj"], test["pair"]),
        "trials": trials,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("selection", "base_test", "raw_test", "selected_test")}, indent=2))


if __name__ == "__main__":
    main()
