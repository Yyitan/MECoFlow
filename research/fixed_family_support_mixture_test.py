"""Protocol-locked test for the diagonal-family/support residual mixture."""

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

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402
from research.residual_mixture_probe import support_delta  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol not preregistered")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    source_paths = [Path(i["path"]).resolve() for i in protocol["sources"]]
    for item, path in zip(protocol["sources"], source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    validation_path = Path(protocol["validation_probe"]["result"]).resolve()
    if sha256(validation_path) != protocol["validation_probe"]["result_sha256"]:
        raise ValueError("validation probe SHA mismatch")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("test_read") is not False:
        raise ValueError("validation probe not test-free")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(p, map_location="cpu", weights_only=True) for p in source_paths]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    vals = torch.stack([zscore(x["val"].float()) for x in saved])
    tests = torch.stack([zscore(x["test"].float()) for x in saved])
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    n = 3
    val_diag = vals[n:].mean(0)
    test_diag = tests[n:].mean(0)
    scfg = protocol["support_config"]
    val_support, _ = support_delta(vals, val_text, cache, scfg)
    test_support, _ = support_delta(tests, test_text, cache, scfg)
    alpha = float(protocol["candidate"]["alpha_diagonal"])
    if abs(alpha + float(protocol["candidate"]["alpha_support"]) - 1.0) > 1e-8:
        raise ValueError("candidate weights do not sum to one")
    val_delta = alpha * 1.25 * val_diag + (1.0 - alpha) * 0.75 * val_support
    test_delta = alpha * 1.25 * test_diag + (1.0 - alpha) * 0.75 * test_support
    val = cache["val"]
    test = cache["test"]
    ev = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    et = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = ev(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = ev(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = et(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = et(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed",
        "test_read": True,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "candidate": protocol["candidate"],
        "base_val": base_val,
        "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test,
        "fused_test": fused_test,
        "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]},
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in source_paths],
        "test_search_after_lock": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain": result["val_gain"], "test_gain": result["test_gain"]}, indent=2))


if __name__ == "__main__":
    main()
