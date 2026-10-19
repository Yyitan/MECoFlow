import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--energies", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def seen_pair_mask(cache: dict[str, object], device: torch.device) -> torch.Tensor:
    meta = cache["meta"]
    mask = torch.zeros(len(meta["pairs"]), dtype=torch.bool, device=device)
    mask[[meta["pairs"].index(pair) for pair in meta["train_pairs"]]] = True
    return mask


def normalize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(1, keepdim=True).clamp_min(1)
    mean = (values * mask).sum(1, keepdim=True) / count
    variance = (((values - mean) * mask) ** 2).sum(1, keepdim=True) / count
    return (values - mean) / variance.sqrt().clamp_min(1e-6)


def adjustment(
    kind: str,
    indices: torch.Tensor,
    energy: torch.Tensor,
    seen_mask: torch.Tensor,
) -> torch.Tensor:
    candidate_seen = seen_mask[indices]
    candidate_unseen = ~candidate_seen
    if kind == "all":
        return normalize(energy, torch.ones_like(candidate_seen))
    if kind == "group":
        return normalize(energy, candidate_seen) * candidate_seen + normalize(energy, candidate_unseen) * candidate_unseen
    if kind == "seen_only":
        return normalize(energy, candidate_seen) * candidate_seen
    if kind == "unseen_only":
        return normalize(energy, candidate_unseen) * candidate_unseen
    raise ValueError(kind)


def fuse(base: torch.Tensor, indices: torch.Tensor, delta: torch.Tensor, strength: float) -> torch.Tensor:
    output = base.clone()
    output.scatter_add_(1, indices, strength * delta)
    return output


def rank_diagnostics(
    split: dict[str, torch.Tensor], indices: torch.Tensor, energy: torch.Tensor, seen_mask: torch.Tensor
) -> dict[str, float]:
    truth = split["pair"].to(indices.device)
    located = indices == truth[:, None]
    covered = located.any(1)
    truth_position = located.float().argmax(1)
    truth_energy = energy[torch.arange(len(energy), device=energy.device), truth_position]
    energy_rank = 1 + (energy > truth_energy[:, None]).sum(1)
    base_top = split["logits"].to(indices.device).gather(1, indices)
    truth_base = base_top[torch.arange(len(base_top), device=base_top.device), truth_position]
    base_rank = 1 + (base_top > truth_base[:, None]).sum(1)
    sample_seen = seen_mask[truth]
    result: dict[str, float] = {"coverage": float(covered.float().mean())}
    for name, mask in {
        "all": covered,
        "seen": covered & sample_seen,
        "unseen": covered & ~sample_seen,
    }.items():
        result[f"energy_top1_{name}"] = float((energy_rank[mask] == 1).float().mean())
        result[f"energy_mrr_{name}"] = float((1.0 / energy_rank[mask].float()).mean())
        result[f"base_top1_{name}"] = float((base_rank[mask] == 1).float().mean())
        result[f"base_mrr_{name}"] = float((1.0 / base_rank[mask].float()).mean())
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = torch.load(args.energies, map_location="cpu", weights_only=False)
    seen_mask = seen_pair_mask(cache, device)
    val_dataset = CompositionDataset(args.dataset_path, phase="val")
    test_dataset = CompositionDataset(args.dataset_path, phase="test")
    val_eval = FastClosedWorldEvaluator(val_dataset, device)
    test_eval = FastClosedWorldEvaluator(test_dataset, device)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_indices, val_energy = saved["val_indices"].to(device), saved["val_energy"].to(device)
    test_indices, test_energy = saved["test_indices"].to(device), saved["test_energy"].to(device)

    result: dict[str, object] = {
        "rank_val": rank_diagnostics(cache["val"], val_indices, val_energy, seen_mask),
        "rank_test": rank_diagnostics(cache["test"], test_indices, test_energy, seen_mask),
    }
    strengths = (-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.35, -0.2, -0.1, -0.05,
                 0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    best = None
    for kind in ("all", "group", "seen_only", "unseen_only"):
        val_delta = adjustment(kind, val_indices, val_energy, seen_mask)
        kind_trials = {}
        for strength in strengths:
            logits = fuse(val["logits"].float(), val_indices, val_delta, strength)
            stats = val_eval(logits, val["attr"], val["obj"], val["pair"])
            kind_trials[f"{strength:.2f}"] = stats
            candidate = (stats["AUC"], stats["best_hm"], kind, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        result[kind] = kind_trials
    assert best is not None
    _, _, kind, strength = best
    test_delta = adjustment(kind, test_indices, test_energy, seen_mask)
    test_logits = fuse(test["logits"].float(), test_indices, test_delta, strength)
    result["selection"] = {"kind": kind, "strength": strength, "val_auc": best[0], "val_hm": best[1]}
    result["fused_test"] = test_eval(test_logits, test["attr"], test["obj"], test["pair"])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "rank_val": result["rank_val"],
        "rank_test": result["rank_test"],
        "selection": result["selection"],
        "fused_test": result["fused_test"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
