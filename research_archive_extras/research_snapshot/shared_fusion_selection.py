"""Select one fusion hyperparameter on mean validation metrics across seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import (  # noqa: E402
    FastClosedWorldEvaluator,
    FastOpenWorldEvaluator,
    zscore,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-val-auc-gain", type=float, default=0.002)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--fixed-delta", choices=("full", "update"))
    parser.add_argument("--fixed-strength", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--protocol",
        help="preregistered JSON whose frozen_fusion block is authoritative",
    )
    args = parser.parse_args()
    selection_source: dict[str, object] = {"type": "validation_search"}
    if args.protocol:
        protocol_path = Path(args.protocol).resolve()
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        frozen = protocol["frozen_fusion"]
        protocol_delta = str(frozen["delta"])
        protocol_strength = float(frozen["strength"])
        protocol_min_gain = float(frozen["min_val_auc_gain"])
        if args.fixed_delta is not None and args.fixed_delta != protocol_delta:
            parser.error("--fixed-delta conflicts with preregistered protocol")
        if args.fixed_strength is not None and args.fixed_strength != protocol_strength:
            parser.error("--fixed-strength conflicts with preregistered protocol")
        if args.min_val_auc_gain != protocol_min_gain:
            parser.error("--min-val-auc-gain conflicts with preregistered protocol")
        args.fixed_delta = protocol_delta
        args.fixed_strength = protocol_strength
        selection_source = {
            "type": "preregistered_protocol",
            "path": str(protocol_path),
            "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "status": protocol.get("status"),
        }
    if (args.fixed_delta is None) != (args.fixed_strength is None):
        parser.error("--fixed-delta and --fixed-strength must be supplied together")
    device = torch.device(args.device)
    cache_path = Path(args.cache).resolve()
    cache_bytes = cache_path.read_bytes()
    cache_record = {
        "path": str(cache_path),
        "bytes": len(cache_bytes),
        "sha256": hashlib.sha256(cache_bytes).hexdigest(),
    }
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    if args.validation_only:
        for index, item in enumerate(saved):
            if set(item) != {"val"}:
                raise ValueError(
                    "validation-only classifier logits must contain exactly the val key; "
                    f"input {index} has keys {sorted(item)}"
                )
    val = {key: value.to(device) for key, value in cache["val"].items()}
    test = None
    if not args.validation_only:
        test = {key: value.to(device) for key, value in cache["test"].items()}
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1).to(device)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = None
    if test is not None:
        test_text = zscore(test["com"].float() @ text.T)
    if cache["meta"].get("open_world", False):
        val_eval = FastOpenWorldEvaluator(cache, device)
        test_eval = None if args.validation_only else FastOpenWorldEvaluator(cache, device)
    else:
        val_eval = FastClosedWorldEvaluator(
            CompositionDataset(args.dataset_path, phase="val"), device
        )
        test_eval = None
        if not args.validation_only:
            test_eval = FastClosedWorldEvaluator(
                CompositionDataset(args.dataset_path, phase="test"), device
            )
    active_mask = torch.tensor(
        cache["meta"].get("active", [True] * text.shape[0]),
        dtype=torch.float32,
        device=device,
    )
    strengths = (
        (args.fixed_strength,)
        if args.fixed_strength is not None
        else (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    )
    delta_names = (args.fixed_delta,) if args.fixed_delta is not None else ("full", "update")
    candidates, best = {}, None
    cached_delta = []
    for item in saved:
        val_full = zscore(item["val"].float().to(device))
        test_full = None
        if not args.validation_only:
            if "test" not in item:
                raise ValueError("test logits missing; rerun without --validation-only")
            test_full = zscore(item["test"].float().to(device))
        cached_delta.append({
            "full": (
                val_full * active_mask[None],
                None if test_full is None else test_full * active_mask[None],
            ),
            "update": (
                (val_full - val_text) * active_mask[None],
                None
                if test_full is None or test_text is None
                else (test_full - test_text) * active_mask[None],
            ),
        })
    for delta_name in delta_names:
        for strength in strengths:
            stats = [
                val_eval(
                    val["logits"].float() + strength * delta[delta_name][0],
                    val["attr"], val["obj"], val["pair"],
                )
                for delta in cached_delta
            ]
            auc, hm = np.mean([x["AUC"] for x in stats]), np.mean([x["best_hm"] for x in stats])
            name = f"{delta_name}_{strength:.2f}"
            candidates[name] = {"mean_val_auc": float(auc), "mean_val_hm": float(hm), "runs": stats}
            candidate = (auc, hm, delta_name, strength)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    val_auc_gain = float(best[0] - base_val["AUC"])
    enabled = val_auc_gain >= args.min_val_auc_gain
    selected_delta = best[2]
    selected_strength = best[3] if enabled else 0.0
    output = {
        "selection": {
            "delta": selected_delta, "strength": selected_strength,
            "raw_strength": best[3],
            "mean_val_auc": float(best[0]), "mean_val_hm": float(best[1]),
            "val_auc_gain": val_auc_gain,
            "min_val_auc_gain": args.min_val_auc_gain,
            "enabled": enabled,
            "source": selection_source,
            "input_cache": cache_record,
        },
        "base_val": base_val,
        "candidates": candidates,
    }
    if not args.validation_only:
        assert test_eval is not None and test is not None
        test_stats = []
        for delta in cached_delta:
            test_delta = delta[selected_delta][1]
            assert test_delta is not None
            test_stats.append(
                test_eval(
                    test["logits"].float() + selected_strength * test_delta,
                    test["attr"], test["obj"], test["pair"],
                )
            )
        std_ddof = 1 if len(test_stats) > 1 else 0
        output["base_test"] = test_eval(
            test["logits"].float(), test["attr"], test["obj"], test["pair"]
        )
        output["mean_test"] = {
            "AUC": float(np.mean([x["AUC"] for x in test_stats])),
            "AUC_std": float(np.std([x["AUC"] for x in test_stats], ddof=std_ddof)),
            "best_hm": float(np.mean([x["best_hm"] for x in test_stats])),
            "best_hm_std": float(
                np.std([x["best_hm"] for x in test_stats], ddof=std_ddof)
            ),
        }
        output["test_runs"] = test_stats
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary_keys = ["selection"]
    if "mean_test" in output:
        summary_keys.extend(["mean_test", "test_runs"])
    print(json.dumps({key: output[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
