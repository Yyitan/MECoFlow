"""Evaluate one frozen support-gated UT candidate after validation gating."""

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


def gate_from_train(cache: dict[str, object]) -> torch.Tensor:
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
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("fixed test requires exactly one candidate")
    candidate = candidates[0]
    if candidate.get("name") != "min_image_support_continuous_full":
        raise ValueError("unexpected candidate")
    if float(candidate.get("strength")) != 0.2:
        raise ValueError("unexpected strength")
    if "no strength" not in protocol.get("test_rule", ""):
        raise ValueError("protocol does not prohibit post-test search")

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Intentional fixed test access follows protocol verification above.
    val, test = cache["val"], cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in args.classifier_logits]
    gate = gate_from_train(cache)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    rows = []
    for path, item in zip(args.classifier_logits, saved):
        val_score = zscore(item["val"].float())
        test_score = zscore(item["test"].float())
        fused_val = val_eval(val["logits"].float() + 0.2 * val_score * gate[None], val["attr"], val["obj"], val["pair"])
        fused_test = test_eval(test["logits"].float() + 0.2 * test_score * gate[None], test["attr"], test["obj"], test["pair"])
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
            "selection_locked": True,
            "test_read": True,
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["mean"], indent=2))


if __name__ == "__main__":
    main()
