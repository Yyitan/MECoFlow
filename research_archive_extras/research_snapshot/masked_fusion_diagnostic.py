"""Validation-selected masking of generated-classifier score updates."""

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
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    real_centroids,
    zscore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    generated = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.classifier_logits
    ]
    _centroids, counts = real_centroids(cache)
    masks = {
        "all": torch.ones_like(counts, dtype=torch.float32, device=device),
        "unseen_only": (counts == 0).float().to(device),
        "seen_only": (counts > 0).float().to(device),
    }
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_generated = torch.stack(
        [zscore(item["val"].float().to(device)) for item in generated]
    ).mean(0)
    test_generated = torch.stack(
        [zscore(item["test"].float().to(device)) for item in generated]
    ).mean(0)
    val_text, test_text = zscore(val["com"].float() @ text.T), zscore(test["com"].float() @ text.T)
    deltas = {
        "full": (val_generated, test_generated),
        "update": (val_generated - val_text, test_generated - test_text),
    }
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    trials = {}
    best = None
    strengths = (-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    for mask_name, mask in masks.items():
        for delta_name, (val_delta, test_delta) in deltas.items():
            for strength in strengths:
                stats = val_eval(
                    val["logits"].float() + strength * val_delta * mask[None],
                    val["attr"], val["obj"], val["pair"],
                )
                name = f"{mask_name}_{delta_name}_{strength:.2f}"
                trials[name] = stats
                candidate = (
                    stats["AUC"] + stats["best_hm"],
                    stats["AUC"],
                    stats["best_hm"],
                    mask_name,
                    delta_name,
                    strength,
                    test_delta,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    assert best is not None
    test_logits = test["logits"].float() + best[5] * best[6] * masks[best[3]][None]
    fused_test = test_eval(test_logits, test["attr"], test["obj"], test["pair"])
    output = {
        "base_val": base_val,
        "base_test": base_test,
        "selection": {
            "mask": best[3],
            "delta": best[4],
            "strength": best[5],
            "selection_score": best[0],
            "val_auc": best[1],
            "val_hm": best[2],
        },
        "fused_test": fused_test,
        "trials": trials,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("base_test", "selection", "fused_test")}, indent=2))


if __name__ == "__main__":
    main()
