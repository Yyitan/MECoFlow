"""Fuse ReCoFlow ensembles while suppressing seed-unstable class evidence."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def ensemble_variants(items: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    stack = torch.stack([zscore(item.float()) for item in items])
    mean = stack.mean(0)
    uncertainty = stack.std(0, unbiased=False)
    # Normalize per image so the gate measures relative class disagreement and
    # does not merely track an image's global logit scale.
    scale = uncertainty.mean(1, keepdim=True).clamp_min(1e-4)
    relative = uncertainty / scale
    variants = {"mean": mean}
    for strength in (0.25, 0.5, 1.0, 2.0, 4.0):
        variants[f"shrink_{strength:g}"] = mean / (1.0 + strength * relative)
        variants[f"exp_{strength:g}"] = mean * torch.exp(-strength * relative)
    for strength in (0.05, 0.1, 0.2, 0.5):
        variants[f"lcb_{strength:g}"] = mean - strength * relative
    return {name: zscore(value) for name, value in variants.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()
    if len(args.classifier_logits) < 2:
        raise ValueError("uncertainty fusion requires at least two generator seeds")

    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_variants = ensemble_variants([item["val"].to(device) for item in saved])
    test_variants = ensemble_variants([item["test"].to(device) for item in saved])
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])

    trials: dict[str, object] = {}
    best = None
    strengths = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5)
    for variant, val_score in val_variants.items():
        for delta_name, val_delta in {
            "full": val_score,
            "update": val_score - val_text,
        }.items():
            for strength in strengths:
                stats = val_eval(
                    val["logits"].float() + strength * val_delta,
                    val["attr"], val["obj"], val["pair"],
                )
                name = f"{variant}_{delta_name}_{strength:g}"
                trials[name] = stats
                candidate = (stats["AUC"], stats["best_hm"], variant, delta_name, strength)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    assert best is not None
    gain = best[0] - base_val["AUC"]
    enabled = gain >= args.min_val_auc_gain
    applied_strength = best[4] if enabled else 0.0
    test_score = test_variants[best[2]]
    test_delta = test_score if best[3] == "full" else test_score - test_text
    raw_test = test_eval(
        test["logits"].float() + best[4] * test_delta,
        test["attr"], test["obj"], test["pair"],
    )
    selected_test = test_eval(
        test["logits"].float() + applied_strength * test_delta,
        test["attr"], test["obj"], test["pair"],
    )
    output = {
        "base_val": base_val,
        "base_test": base_test,
        "selection": {
            "variant": best[2], "delta": best[3],
            "strength": applied_strength, "raw_strength": best[4],
            "val_auc": best[0], "val_hm": best[1], "val_auc_gain": gain,
            "min_val_auc_gain": args.min_val_auc_gain, "enabled": enabled,
        },
        "raw_test": raw_test,
        "selected_test": selected_test,
        "trials": trials,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("selection", "base_test", "raw_test", "selected_test")}, indent=2))


if __name__ == "__main__":
    main()
