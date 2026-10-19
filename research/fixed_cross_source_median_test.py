"""Run one protocol-locked six-source median test evaluation."""

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

from dataset import CompositionDataset
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(value: str, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol status is not fixed before test")
    candidates = protocol.get("candidate_set")
    if len(candidates) != 1:
        raise ValueError("fixed test requires exactly one candidate")
    candidate = candidates[0]
    if candidate.get("name") != "ut_cross_source_median_update" or float(candidate.get("strength")) != 1.25:
        raise ValueError("unexpected fixed candidate")
    if "do not search strength" not in protocol.get("test_rule", ""):
        raise ValueError("protocol does not prohibit post-test search")

    cache_path = resolve(protocol["cache"], root)
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA256 mismatch")
    source_paths = []
    for item in protocol["sources"]:
        path = resolve(item["path"], root)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA256 mismatch: {path}")
        source_paths.append(path)

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_scores = [zscore(item["val"].float()) for item in saved]
    test_scores = [zscore(item["test"].float()) for item in saved]
    val_score = torch.stack(val_scores).median(0).values
    test_score = torch.stack(test_scores).median(0).values
    val_delta, test_delta = val_score - val_text, test_score - test_text

    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = 1.25
    fused_val = val_eval(val["logits"].float() + strength * val_delta, val["attr"], val["obj"], val["pair"])
    fused_test = test_eval(test["logits"].float() + strength * test_delta, test["attr"], test["obj"], test["pair"])

    individual = []
    for index, (val_item, test_item) in enumerate(zip(val_scores, test_scores)):
        individual_val = val_eval(val["logits"].float() + strength * (val_item - val_text), val["attr"], val["obj"], val["pair"])
        individual_test = test_eval(test["logits"].float() + strength * (test_item - test_text), test["attr"], test["obj"], test["pair"])
        individual.append({
            "seed_index": index,
            "val_auc_gain": individual_val["AUC"] - base_val["AUC"],
            "val_hm_gain": individual_val["best_hm"] - base_val["best_hm"],
            "test_auc_gain": individual_test["AUC"] - base_test["AUC"],
            "test_hm_gain": individual_test["best_hm"] - base_test["best_hm"],
            "fused_val": individual_val,
            "fused_test": individual_test,
        })
    result = {
        "status": "passed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256(protocol_path),
            "candidate": candidate,
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
            "selection_locked": True,
            "test_read": True,
        },
        "base_val": base_val,
        "fused_val": fused_val,
        "base_test": base_test,
        "fused_test": fused_test,
        "individual": individual,
        "mean": {
            "val_auc_gain": fused_val["AUC"] - base_val["AUC"],
            "val_hm_gain": fused_val["best_hm"] - base_val["best_hm"],
            "test_auc_gain": fused_test["AUC"] - base_test["AUC"],
            "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"],
            "individual_test_auc_gain_mean": float(np.mean([r["test_auc_gain"] for r in individual])),
            "individual_test_auc_gain_std": float(np.std([r["test_auc_gain"] for r in individual])),
            "individual_test_hm_gain_mean": float(np.mean([r["test_hm_gain"] for r in individual])),
            "individual_test_hm_gain_std": float(np.std([r["test_hm_gain"] for r in individual])),
            "positive_individual_test_auc_runs": sum(r["test_auc_gain"] > 0 for r in individual),
            "positive_individual_test_hm_runs": sum(r["test_hm_gain"] > 0 for r in individual),
        },
        "test_search_after_lock": False,
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
