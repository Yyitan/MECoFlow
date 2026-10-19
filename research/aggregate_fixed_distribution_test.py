"""Evaluate a pre-locked source ensemble across independently generated logits."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    frozen = protocol.get("frozen_fusion", {})
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered before fixed test")
    if frozen.get("delta") != "update" or float(frozen.get("strength")) != 0.75:
        raise ValueError("unexpected frozen candidate")
    if "no post-test search" not in protocol.get("test_rule", ""):
        raise ValueError("protocol does not prohibit post-test search")

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(p, map_location="cpu", weights_only=True) for p in args.classifier_logits]
    if any(set(item) != {"val", "test"} for item in saved):
        raise ValueError("fixed test requires exactly {'val','test'} logits")
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_scores = [zscore(item["val"].float()) for item in saved]
    test_scores = [zscore(item["test"].float()) for item in saved]
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = 0.75
    ensemble_val = val_eval(val["logits"].float() + strength * (torch.stack(val_scores).mean(0) - val_text), val["attr"], val["obj"], val["pair"])
    ensemble_test = test_eval(test["logits"].float() + strength * (torch.stack(test_scores).mean(0) - test_text), test["attr"], test["obj"], test["pair"])
    individual = []
    for index, (vscore, tscore) in enumerate(zip(val_scores, test_scores)):
        fv = val_eval(val["logits"].float() + strength * (vscore - val_text), val["attr"], val["obj"], val["pair"])
        ft = test_eval(test["logits"].float() + strength * (tscore - test_text), test["attr"], test["obj"], test["pair"])
        individual.append({
            "seed_index": index,
            "fused_val": fv,
            "val_auc_gain": fv["AUC"] - base_val["AUC"],
            "val_hm_gain": fv["best_hm"] - base_val["best_hm"],
            "fused_test": ft,
            "test_auc_gain": ft["AUC"] - base_test["AUC"],
            "test_hm_gain": ft["best_hm"] - base_test["best_hm"],
        })
    result = {
        "status": "passed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256(protocol_path),
            "cache": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "frozen_fusion": frozen,
            "classifier_logits": [{"path": str(Path(p).resolve()), "sha256": sha256(Path(p).resolve())} for p in args.classifier_logits],
            "selection_locked": True,
            "test_read": True,
        },
        "base_val": base_val,
        "ensemble_val": ensemble_val,
        "base_test": base_test,
        "ensemble_test": ensemble_test,
        "individual": individual,
        "mean": {
            "ensemble_val_auc_gain": ensemble_val["AUC"] - base_val["AUC"],
            "ensemble_val_hm_gain": ensemble_val["best_hm"] - base_val["best_hm"],
            "ensemble_test_auc_gain": ensemble_test["AUC"] - base_test["AUC"],
            "ensemble_test_hm_gain": ensemble_test["best_hm"] - base_test["best_hm"],
            "individual_test_auc_gain_mean": float(np.mean([r["test_auc_gain"] for r in individual])),
            "individual_test_auc_gain_std": float(np.std([r["test_auc_gain"] for r in individual])),
            "individual_test_hm_gain_mean": float(np.mean([r["test_hm_gain"] for r in individual])),
            "individual_test_hm_gain_std": float(np.std([r["test_hm_gain"] for r in individual])),
            "positive_individual_test_auc_runs": int(sum(r["test_auc_gain"] > 0 for r in individual)),
            "positive_individual_test_hm_runs": int(sum(r["test_hm_gain"] > 0 for r in individual)),
        },
        "test_search_after_lock": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
