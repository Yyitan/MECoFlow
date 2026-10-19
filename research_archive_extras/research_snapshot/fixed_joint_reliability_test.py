"""Protocol-locked fixed test for a joint support/confidence transport gate."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def support(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    attr_index = {value: index for index, value in enumerate(meta["attrs"])}
    obj_index = {value: index for index, value in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attr_index[attr] for attr, _ in meta["pairs"]])
    pair_obj = torch.tensor([obj_index[obj] for _, obj in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attr_index)).float()[pair_attr]
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(obj_index)).float()[pair_obj]
    if mode == "min":
        raw = torch.minimum(attr_count, obj_count)
    elif mode == "harmonic":
        raw = 2 * attr_count * obj_count / (attr_count + obj_count).clamp_min(1e-6)
    elif mode == "object":
        raw = obj_count
    elif mode == "attribute":
        raw = attr_count
    else:
        raise ValueError(f"unknown support mode: {mode}")
    return normalize(torch.log1p(raw))


def confidence_gate(logits: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(logits, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
    values = {
        "entropy_low": 1 - normalize(entropy),
        "entropy_high": normalize(entropy),
        "margin_low": 1 - normalize(margin),
        "margin_high": normalize(margin),
    }
    return values[name]


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


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
        raise ValueError("protocol must be preregistered")
    candidates = protocol.get("candidate_set", [])
    if len(candidates) != 1:
        raise ValueError("exactly one candidate is required")
    candidate = candidates[0]
    required = {"support", "confidence", "support_power", "confidence_power", "combine", "strength"}
    if set(candidate) != required:
        raise ValueError(f"candidate keys mismatch: {sorted(candidate)}")
    if "validation AUC gain" not in protocol.get("validation_selection_rule", ""):
        raise ValueError("validation selection rule missing AUC gate")
    if "largest validation HM" not in protocol.get("validation_selection_rule", ""):
        raise ValueError("validation selection rule missing HM lock")
    if "do not search" not in protocol.get("test_rule", ""):
        raise ValueError("test rule does not prohibit search")
    if protocol.get("validation_gate", {}).get("test_read") is not False:
        raise ValueError("validation gate must be test-free")

    cache_path = resolve(protocol["cache"], root)
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache sha mismatch")
    source_paths = [resolve(item["path"], root) for item in protocol["sources"]]
    for path, item in zip(source_paths, protocol["sources"]):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source sha mismatch: {path}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_base_logits = val["logits"].float()
    test_base_logits = test["logits"].float()
    val_base_z = zscore(val_base_logits)
    test_base_z = zscore(test_base_logits)
    val_source = torch.stack([zscore(item["val"].float()) for item in saved]).median(0).values
    test_source = torch.stack([zscore(item["test"].float()) for item in saved]).median(0).values

    support_value = support(cache, candidate["support"])
    val_confidence = confidence_gate(val_base_z, candidate["confidence"])
    test_confidence = confidence_gate(test_base_z, candidate["confidence"])
    support_value = support_value.clamp_min(1e-6).pow(float(candidate["support_power"]))
    val_confidence = val_confidence.clamp_min(1e-6).pow(float(candidate["confidence_power"]))
    test_confidence = test_confidence.clamp_min(1e-6).pow(float(candidate["confidence_power"]))
    if candidate["combine"] == "product":
        val_gate = support_value[None] * val_confidence[:, None]
        test_gate = support_value[None] * test_confidence[:, None]
    elif candidate["combine"] == "linear":
        val_gate = 0.5 * support_value[None] + 0.5 * val_confidence[:, None]
        test_gate = 0.5 * support_value[None] + 0.5 * test_confidence[:, None]
    else:
        raise ValueError(f"unknown combine mode: {candidate['combine']}")
    val_gate = val_gate / val_gate.mean().clamp_min(1e-6)
    test_gate = test_gate / test_gate.mean().clamp_min(1e-6)
    strength = float(candidate["strength"])

    val_evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_evaluator(val_base_logits, val["attr"], val["obj"], val["pair"])
    base_test = test_evaluator(test_base_logits, test["attr"], test["obj"], test["pair"])
    fused_val = val_evaluator(val_base_logits + strength * val_gate * (val_source - val_text), val["attr"], val["obj"], val["pair"])
    fused_test = test_evaluator(test_base_logits + strength * test_gate * (test_source - test_text), test["attr"], test["obj"], test["pair"])

    expected = protocol["validation_gate"]
    val_auc_gain = fused_val["AUC"] - base_val["AUC"]
    val_hm_gain = fused_val["best_hm"] - base_val["best_hm"]
    if abs(val_auc_gain - float(expected["mean_auc_gain"])) > 1e-10:
        raise ValueError(f"validation AUC replay mismatch: {val_auc_gain} vs {expected['mean_auc_gain']}")
    if abs(val_hm_gain - float(expected["mean_hm_gain"])) > 1e-10:
        raise ValueError(f"validation HM replay mismatch: {val_hm_gain} vs {expected['mean_hm_gain']}")

    individual = []
    for index, item in enumerate(saved):
        val_one = val_evaluator(val_base_logits + strength * val_gate * (zscore(item["val"].float()) - val_text), val["attr"], val["obj"], val["pair"])
        test_one = test_evaluator(test_base_logits + strength * test_gate * (zscore(item["test"].float()) - test_text), test["attr"], test["obj"], test["pair"])
        individual.append({
            "seed_index": index,
            "val_auc_gain": val_one["AUC"] - base_val["AUC"],
            "val_hm_gain": val_one["best_hm"] - base_val["best_hm"],
            "test_auc_gain": test_one["AUC"] - base_test["AUC"],
            "test_hm_gain": test_one["best_hm"] - base_test["best_hm"],
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
            "val_auc_gain": val_auc_gain,
            "val_hm_gain": val_hm_gain,
            "test_auc_gain": fused_test["AUC"] - base_test["AUC"],
            "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"],
            "positive_individual_test_auc_runs": sum(row["test_auc_gain"] > 0 for row in individual),
            "positive_individual_test_hm_runs": sum(row["test_hm_gain"] > 0 for row in individual),
        },
        "test_search_after_lock": False,
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
