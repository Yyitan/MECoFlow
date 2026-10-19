"""Graph-shrink generated classifier scores through shared CZSL primitives."""

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


def pair_maps(cache: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {value: index for index, value in enumerate(meta["attrs"])}
    objs = {value: index for index, value in enumerate(meta["objs"])}
    return (
        torch.tensor([attrs[a] for a, _ in meta["pairs"]], device=device),
        torch.tensor([objs[o] for _, o in meta["pairs"]], device=device),
    )


def primitive_projection(
    scores: torch.Tensor,
    pair_attr: torch.Tensor,
    pair_obj: torch.Tensor,
    num_attrs: int,
    num_objs: int,
) -> torch.Tensor:
    """Project pair scores onto additive attribute/object graph effects."""
    output = []
    attr_count = torch.bincount(pair_attr, minlength=num_attrs).float().clamp_min(1)
    obj_count = torch.bincount(pair_obj, minlength=num_objs).float().clamp_min(1)
    for chunk in scores.split(512):
        attr_sum = torch.zeros(len(chunk), num_attrs, device=scores.device)
        obj_sum = torch.zeros(len(chunk), num_objs, device=scores.device)
        attr_sum.index_add_(1, pair_attr, chunk)
        obj_sum.index_add_(1, pair_obj, chunk)
        attr_mean = attr_sum / attr_count[None]
        obj_mean = obj_sum / obj_count[None]
        global_mean = chunk.mean(1, keepdim=True)
        output.append(attr_mean[:, pair_attr] + obj_mean[:, pair_obj] - global_mean)
    return torch.cat(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--dataset-path", required=True)
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
    val_update = val_classifier - zscore(val["com"].float() @ text.T)
    test_update = test_classifier - zscore(test["com"].float() @ text.T)
    pair_attr, pair_obj = pair_maps(cache, device)
    val_factor = primitive_projection(
        val_update, pair_attr, pair_obj, len(cache["meta"]["attrs"]), len(cache["meta"]["objs"])
    )
    test_factor = primitive_projection(
        test_update, pair_attr, pair_obj, len(cache["meta"]["attrs"]), len(cache["meta"]["objs"])
    )
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    trials, best = {}, None
    for raw_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        val_delta = raw_weight * val_update + (1.0 - raw_weight) * val_factor
        test_delta = raw_weight * test_update + (1.0 - raw_weight) * test_factor
        val_delta, test_delta = zscore(val_delta), zscore(test_delta)
        for strength in (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            stats = val_eval(
                val["logits"].float() + strength * val_delta,
                val["attr"], val["obj"], val["pair"],
            )
            name = f"raw_{raw_weight:.2f}_strength_{strength:.2f}"
            trials[name] = stats
            candidate = (
                stats["AUC"] + stats["best_hm"], stats["AUC"], stats["best_hm"],
                raw_weight, strength, test_delta,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    assert best is not None
    fused_test = test_eval(
        test["logits"].float() + best[4] * best[5],
        test["attr"], test["obj"], test["pair"],
    )
    result = {
        "base_val": base_val,
        "base_test": base_test,
        "selection": {
            "raw_weight": best[3], "strength": best[4],
            "val_auc": best[1], "val_hm": best[2],
        },
        "fused_test": fused_test,
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("base_test", "selection", "fused_test")}, indent=2))


if __name__ == "__main__":
    main()
