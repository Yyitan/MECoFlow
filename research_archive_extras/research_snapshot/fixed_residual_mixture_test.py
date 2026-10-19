"""Run one protocol-locked test evaluation for residual mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore
from research.residual_mixture_probe import adaptive_delta, support_delta
from dataset import CompositionDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered_before_fixed_test")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    source_paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    probe_path = Path(protocol["validation_probe"]["result"]).resolve()
    if sha256(probe_path) != protocol["validation_probe"]["result_sha256"]:
        raise ValueError("validation probe SHA mismatch")
    if protocol["validation_probe"]["test_read"] is not False:
        raise ValueError("validation probe is not marked test_read=false")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    sources = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["test"].float())
        for path in source_paths
    ])
    val_sources = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in source_paths
    ])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val = cache["val"]
    test = cache["test"]
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    acfg = protocol["components"]["adaptive"]["config"]
    scfg = protocol["components"]["support"]["config"]
    astr = float(protocol["components"]["adaptive"]["strength"])
    sstr = float(protocol["components"]["support"]["strength"])
    alpha = float(protocol["candidate"]["alpha_adaptive"])
    if abs(alpha + float(protocol["candidate"]["alpha_support"]) - 1.0) > 1e-8:
        raise ValueError("mixture coefficients do not sum to one")
    val_adaptive, _ = adaptive_delta(val_sources, val_text, acfg)
    val_support, _ = support_delta(val_sources, val_text, cache, scfg)
    test_adaptive, _ = adaptive_delta(sources, test_text, acfg)
    test_support, _ = support_delta(sources, test_text, cache, scfg)
    val_delta = alpha * astr * val_adaptive + (1.0 - alpha) * sstr * val_support
    test_delta = alpha * astr * test_adaptive + (1.0 - alpha) * sstr * test_support
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "mode": "fixed_test_locked_reliability_residual_mixture",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "test_read": True,
        "candidate": protocol["candidate"],
        "base_val": base_val,
        "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test,
        "fused_test": fused_test,
        "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain": result["val_gain"], "test_gain": result["test_gain"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
