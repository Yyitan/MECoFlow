"""Gate generated CZSL scores by support in the seen composition graph."""

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


def harmonic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 2.0 * left * right / (left + right).clamp_min(1e-6)


def normalize(score: torch.Tensor) -> torch.Tensor:
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    args = parser.parse_args()
    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in args.classifier_logits
    ]
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = {key: value.to(device) for key, value in cache["test"].items()}
    meta = cache["meta"]
    attr_index = {value: index for index, value in enumerate(meta["attrs"])}
    obj_index = {value: index for index, value in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attr_index[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([obj_index[o] for _, o in meta["pairs"]])
    _, pair_counts = real_centroids(cache)
    seen = pair_counts > 0
    attr_pair_degree = torch.bincount(
        pair_attr[seen], minlength=len(meta["attrs"])
    ).float()
    obj_pair_degree = torch.bincount(
        pair_obj[seen], minlength=len(meta["objs"])
    ).float()
    train_pair = cache["train"]["pair"].long()
    attr_image_support = torch.bincount(
        pair_attr[train_pair], minlength=len(meta["attrs"])
    ).float()
    obj_image_support = torch.bincount(
        pair_obj[train_pair], minlength=len(meta["objs"])
    ).float()
    reliability = {
        "min_pair_degree": torch.minimum(
            attr_pair_degree[pair_attr], obj_pair_degree[pair_obj]
        ),
        "harmonic_pair_degree": harmonic(
            attr_pair_degree[pair_attr], obj_pair_degree[pair_obj]
        ),
        "min_image_support": torch.minimum(
            attr_image_support[pair_attr], obj_image_support[pair_obj]
        ),
        "harmonic_image_support": harmonic(
            attr_image_support[pair_attr], obj_image_support[pair_obj]
        ),
    }
    masks = {"all": torch.ones(len(pair_attr), device=device)}
    unseen = ~seen
    for name, raw_score in reliability.items():
        score = normalize(torch.log1p(raw_score)).to(device)
        masks[f"{name}_continuous"] = score
        masks[f"{name}_inverse"] = 1.0 - score
        for keep_fraction in (0.25, 0.5, 0.75):
            unseen_values = raw_score[unseen]
            high = torch.quantile(unseen_values, 1.0 - keep_fraction)
            low = torch.quantile(unseen_values, keep_fraction)
            masks[f"{name}_top{keep_fraction:g}"] = (
                unseen & (raw_score >= high)
            ).float().to(device)
            masks[f"{name}_bottom{keep_fraction:g}"] = (
                unseen & (raw_score <= low)
            ).float().to(device)

    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_generated = torch.stack(
        [zscore(item["val"].float().to(device)) for item in saved]
    ).mean(0)
    test_generated = torch.stack(
        [zscore(item["test"].float().to(device)) for item in saved]
    ).mean(0)
    deltas = {
        "full": (val_generated, test_generated),
        "update": (val_generated - val_text, test_generated - test_text),
    }
    val_eval = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), device
    )
    test_eval = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="test"), device
    )
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strengths = (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    trials, best = {}, None
    for mask_name, mask in masks.items():
        for delta_name, (val_delta, test_delta) in deltas.items():
            for strength in strengths:
                stats = val_eval(
                    val["logits"].float() + strength * val_delta * mask[None],
                    val["attr"], val["obj"], val["pair"],
                )
                name = f"{mask_name}_{delta_name}_{strength:g}"
                trials[name] = stats
                candidate = (
                    stats["AUC"], stats["best_hm"], name,
                    mask_name, delta_name, strength, test_delta,
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    assert best is not None
    val_gain = best[0] - base_val["AUC"]
    enabled = val_gain >= args.min_val_auc_gain
    selected_strength = best[5] if enabled else 0.0
    raw_test = test_eval(
        test["logits"].float() + best[5] * best[6] * masks[best[3]][None],
        test["attr"], test["obj"], test["pair"],
    )
    selected_test = test_eval(
        test["logits"].float()
        + selected_strength * best[6] * masks[best[3]][None],
        test["attr"], test["obj"], test["pair"],
    )
    output = {
        "base_val": base_val, "base_test": base_test,
        "selection": {
            "name": best[2], "mask": best[3], "delta": best[4],
            "strength": selected_strength, "raw_strength": best[5],
            "val_auc": best[0], "val_hm": best[1], "val_auc_gain": val_gain,
            "min_val_auc_gain": args.min_val_auc_gain, "enabled": enabled,
        },
        "raw_test": raw_test, "selected_test": selected_test, "trials": trials,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selection": output["selection"], "raw_test": raw_test, "selected_test": selected_test}, indent=2))


if __name__ == "__main__":
    main()
