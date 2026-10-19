"""Apply a validation-locked reliability gate to the held-out test split."""

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
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator

    selection_document = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    selection = selection_document["selection"]
    device = torch.device(args.device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = row_zscore(val["com"].float() @ text.T)
    test_text = row_zscore(test["com"].float() @ text.T)
    val_base_z = row_zscore(val["logits"].float())
    test_base_z = row_zscore(test["logits"].float())
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="test"), device
    )
    base_stats = evaluator(
        test["logits"].float(), test["attr"], test["obj"], test["pair"]
    )

    run_stats: list[dict[str, float]] = []
    gate_summaries: list[dict[str, float]] = []
    for path in args.classifier_logits:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        val_generated = row_zscore(saved["val"].float().to(device))
        test_generated = row_zscore(saved["test"].float().to(device))
        val_raw = raw_reliability_features(val_base_z, val_generated, val_text)
        val_feature_stats = fit_reliability_stats(val_raw)
        val_features = reliability_features(val_raw, val_feature_stats)
        test_features = reliability_features(
            raw_reliability_features(test_base_z, test_generated, test_text),
            val_feature_stats,
        )
        val_feature = val_features[selection["feature"]]
        test_feature = test_features[selection["feature"]]
        threshold = torch.quantile(val_feature, selection["threshold_quantile"])
        q25, q75 = torch.quantile(
            val_feature, torch.tensor([0.25, 0.75], device=device)
        )
        temperature = (
            (q75 - q25).clamp_min(1e-6) * selection["temperature_scale"]
        )
        gate = torch.sigmoid((test_feature - threshold) / temperature)
        delta = (
            test_generated
            if selection["delta"] == "full"
            else test_generated - test_text
        )
        logits = test["logits"].float() + selection["strength"] * gate[:, None] * delta
        run_stats.append(evaluator(logits, test["attr"], test["obj"], test["pair"]))
        gate_summaries.append(
            {
                "mean": float(gate.mean()),
                "std": float(gate.std()),
                "active_above_half": float((gate > 0.5).float().mean()),
            }
        )

    output = {
        "protocol": "test_application_of_frozen_validation_selection",
        "selection": selection,
        "base_test": base_stats,
        "mean_test": {
            "AUC": float(np.mean([item["AUC"] for item in run_stats])),
            "AUC_std": float(np.std([item["AUC"] for item in run_stats], ddof=1)),
            "best_hm": float(np.mean([item["best_hm"] for item in run_stats])),
            "best_hm_std": float(
                np.std([item["best_hm"] for item in run_stats], ddof=1)
            ),
        },
        "test_runs": run_stats,
        "gate_summaries": gate_summaries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
