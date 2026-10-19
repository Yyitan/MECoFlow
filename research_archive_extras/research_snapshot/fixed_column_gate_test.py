"""Evaluate one frozen C-GQA column-centered support-gate candidate.

The protocol is checked before any test tensors are accessed. There is no
post-test search: the unique candidate and strength come from the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support_gate(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    support = torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj]))
    return (support - support.min()) / (support.max() - support.min()).clamp_min(1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidates = protocol.get("candidate_set")
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol status is not fixed before test")
    if protocol.get("test_rule", "").find("no strength") < 0:
        raise ValueError("protocol does not prohibit post-test search")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("fixed test requires exactly one candidate")
    candidate = candidates[0]
    if candidate.get("name") != "column_centered_min_image_support_full":
        raise ValueError("unexpected candidate")
    if float(candidate.get("strength")) != 0.2:
        raise ValueError("unexpected strength")

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Protocol checks are complete. Test access below is intentional and fixed.
    val = cache["val"]
    test = cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    gate = support_gate(cache)
    evaluator_val = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    evaluator_test = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu")
    )
    base_val = evaluator_val(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = evaluator_test(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    rows = []
    for path, item in zip(args.classifier_logits, saved):
        val_score = zscore(item["val"].float())
        column_mean = val_score.mean(0, keepdim=True)
        test_score = zscore(item["test"].float() - column_mean)
        val_score = zscore(val_score - column_mean)
        fused_val = evaluator_val(
            val["logits"].float() + 0.2 * val_score * gate[None],
            val["attr"], val["obj"], val["pair"],
        )
        fused_test = evaluator_test(
            test["logits"].float() + 0.2 * test_score * gate[None],
            test["attr"], test["obj"], test["pair"],
        )
        rows.append({
            "path": str(Path(path).resolve()),
            "sha256": sha256(Path(path).resolve()),
            "fused_val": fused_val,
            "val_auc_gain": fused_val["AUC"] - base_val["AUC"],
            "val_hm_gain": fused_val["best_hm"] - base_val["best_hm"],
            "fused_test": fused_test,
            "test_auc_gain": fused_test["AUC"] - base_test["AUC"],
            "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"],
        })
    output = {
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256(protocol_path),
            "candidate": candidate,
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "test_read": True,
            "selection_locked": True,
        },
        "base_val": base_val,
        "base_test": base_test,
        "runs": rows,
        "mean": {
            "val_auc_gain": float(np.mean([r["val_auc_gain"] for r in rows])),
            "val_hm_gain": float(np.mean([r["val_hm_gain"] for r in rows])),
            "test_auc_gain": float(np.mean([r["test_auc_gain"] for r in rows])),
            "test_hm_gain": float(np.mean([r["test_hm_gain"] for r in rows])),
            "positive_test_auc_runs": int(sum(r["test_auc_gain"] > 0 for r in rows)),
            "positive_test_hm_runs": int(sum(r["test_hm_gain"] > 0 for r in rows)),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["mean"], indent=2))


if __name__ == "__main__":
    main()
