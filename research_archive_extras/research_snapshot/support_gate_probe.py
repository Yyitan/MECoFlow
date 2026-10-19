"""CPU-only validation probe for support/consistency gated score updates.

This exploratory script deliberately reads only cache['val'] and the ``val``
branch of generator logits. It is not a formal selection protocol; it is used
to decide whether a pair-level reliability gate merits a preregistered replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids, zscore


def pair_support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    support = torch.minimum(attr_count[pair_attr], obj_count[pair_obj])
    score = torch.log1p(support)
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(torch.get_num_threads(), 16))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    # Do not touch cache['test']; this is intentionally validation-only.
    val = cache["val"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    val_scores = [zscore(item["val"].float()) for item in saved]
    val_base = val["logits"].float()
    support = pair_support(cache)
    seen_pair = torch.bincount(cache["train"]["pair"].long(), minlength=len(support)) > 0

    # Pair-level generator disagreement. Lower disagreement means that a pair's
    # generated evidence is stable across stochastic replay seeds.
    stack = torch.stack(val_scores)
    dispersion = stack.std(0, unbiased=False).mean(0)
    median = dispersion.median().clamp_min(1e-6)
    consistency = (1.0 / (1.0 + dispersion / median)).clamp(0.0, 1.0)
    gates = {
        "support": support,
        "support_consistency": support * consistency,
        "support_consistency_sqrt": support * consistency.sqrt(),
        # Asymmetric variants test whether generated evidence should correct
        # unseen pairs more aggressively while preserving seen calibration.
        "support_unseen_only": support * (~seen_pair).float(),
        "support_seen_only": support * seen_pair.float(),
        "support_unseen_boost": support * (1.0 + (~seen_pair).float()),
    }
    dataset = CompositionDataset(args.dataset_path, phase="val")
    evaluator = FastClosedWorldEvaluator(dataset, torch.device("cpu"))
    base = evaluator(val_base, val["attr"], val["obj"], val["pair"])
    strengths = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    score_variants = {
        "raw": val_scores,
        # Unsupervised pair-prior calibration: remove a pair's mean generated
        # evidence before row-wise normalization. This tests whether the flow
        # introduces class-dependent score bias rather than useful evidence.
        "column_centered": [zscore(score - score.mean(0, keepdim=True)) for score in val_scores],
    }
    rows = []
    for variant_name, variant_scores in score_variants.items():
        for gate_name, gate in gates.items():
            for strength in strengths:
                gains = []
                hm_gains = []
                for score in variant_scores:
                    fused = val_base + strength * score * gate[None]
                    stats = evaluator(fused, val["attr"], val["obj"], val["pair"])
                    gains.append(stats["AUC"] - base["AUC"])
                    hm_gains.append(stats["best_hm"] - base["best_hm"])
                rows.append({
                    "variant": variant_name,
                    "gate": gate_name,
                    "strength": strength,
                    "mean_auc_gain": float(np.mean(gains)),
                    "mean_hm_gain": float(np.mean(hm_gains)),
                    "auc_gains": gains,
                    "hm_gains": hm_gains,
                    "positive_auc_runs": int(sum(x > 0 for x in gains)),
                    "positive_hm_runs": int(sum(x > 0 for x in hm_gains)),
                })
    rows.sort(key=lambda r: (r["mean_auc_gain"], r["mean_hm_gain"]), reverse=True)
    output = {
        "mode": "exploratory_validation_only",
        "test_read": False,
        "cache": str(Path(args.cache).resolve()),
        "classifier_logits": [str(Path(p).resolve()) for p in args.classifier_logits],
        "base_val": base,
        "rows": rows,
        "best": rows[0],
        "definition": "pair_support multiplied by pair-level cross-seed consistency",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"base_val": base, "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
