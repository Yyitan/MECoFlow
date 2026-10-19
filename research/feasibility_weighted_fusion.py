"""Continuous feasibility weighting for generated open-world score updates."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.fast_feature_evaluation import FastOpenWorldEvaluator, zscore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--feasibility", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    val_classifier = torch.stack([zscore(item["val"].float().to(device)) for item in saved]).mean(0)
    test_classifier = torch.stack([zscore(item["test"].float().to(device)) for item in saved]).mean(0)
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    deltas = {
        "full": (val_classifier, test_classifier),
        "update": (val_classifier - val_text, test_classifier - test_text),
    }

    meta = cache["meta"]
    full_pairs = list(itertools.product(meta["attrs"], meta["objs"]))
    full_index = {pair: index for index, pair in enumerate(full_pairs)}
    feasibility = torch.load(args.feasibility, map_location="cpu", weights_only=True)["feasibility"]
    candidate_feasibility = torch.tensor(
        [feasibility[full_index[tuple(pair)]] for pair in meta["pairs"]], device=device
    )
    seen_pairs = set(map(tuple, meta["train_pairs"]))
    seen = torch.tensor([tuple(pair) in seen_pairs for pair in meta["pairs"]], device=device)
    active = torch.tensor(meta["active"], dtype=torch.bool, device=device)
    normalized = ((candidate_feasibility - args.threshold) / (feasibility.max() - args.threshold)).clamp(0, 1)

    evaluator = FastOpenWorldEvaluator(cache, device)
    base_val = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = evaluator(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    trials, best = {}, None
    for exponent in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        gate = active.float() if exponent == 0 else normalized.pow(exponent) * active.float()
        gate[seen] = 1.0
        for delta_name, (val_delta, test_delta) in deltas.items():
            for strength in (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
                stats = evaluator(
                    val["logits"].float() + strength * val_delta * gate[None],
                    val["attr"], val["obj"], val["pair"],
                )
                name = f"exponent_{exponent:g}_{delta_name}_{strength:.2f}"
                trials[name] = stats
                candidate = (
                    stats["AUC"] + stats["best_hm"], stats["AUC"], stats["best_hm"],
                    exponent, delta_name, strength, test_delta, gate,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    assert best is not None
    fused_test = evaluator(
        test["logits"].float() + best[5] * best[6] * best[7][None],
        test["attr"], test["obj"], test["pair"],
    )
    result = {
        "base_val": base_val,
        "base_test": base_test,
        "selection": {
            "exponent": best[3], "delta": best[4], "strength": best[5],
            "val_auc": best[1], "val_hm": best[2],
        },
        "fused_test": fused_test,
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("base_test", "selection", "fused_test")}, indent=2))


if __name__ == "__main__":
    main()
