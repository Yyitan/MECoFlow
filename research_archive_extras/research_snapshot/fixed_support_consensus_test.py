"""Run one protocol-locked test for support-aware consensus fusion."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def support_gate(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()[pair_attr]
    obj = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()[pair_obj]
    if mode == "harmonic_image_support":
        raw = 2.0 * attr * obj / (attr + obj).clamp_min(1e-6)
    elif mode == "min_image_support":
        raw = torch.minimum(attr, obj)
    elif mode == "object_image_support":
        raw = obj
    else:
        raise ValueError(f"unsupported support mode: {mode}")
    return (torch.log1p(raw) - torch.log1p(raw).min()) / (
        torch.log1p(raw).max() - torch.log1p(raw).min()
    ).clamp_min(1e-6)


def gated_delta(scores: torch.Tensor, text_logits: torch.Tensor, candidate: dict[str, object], cache: dict[str, object]) -> torch.Tensor:
    aggregation = candidate.get("aggregation", "median")
    center = scores.mean(0) if aggregation == "mean" else scores.median(0).values
    dispersion = (scores - center).abs().mean(0)
    relative = dispersion / dispersion.median(1, keepdim=True).values.clamp_min(1e-6)
    tau = float(candidate["tau"])
    if candidate["consensus"] == "none":
        agreement = torch.ones_like(relative)
    elif candidate["consensus"] == "inverse":
        agreement = 1.0 / (1.0 + tau * relative)
    elif candidate["consensus"] == "exp":
        agreement = torch.exp(-tau * relative)
    else:
        raise ValueError("unsupported consensus")
    agreement = agreement / agreement.mean(1, keepdim=True).clamp_min(1e-6)
    support = support_gate(cache, str(candidate["support_mode"]))
    support_power = support.clamp_min(1e-6).pow(float(candidate["support_exponent"]))
    if candidate["blend"] == "linear":
        gate = 0.5 * agreement + 0.5 * support_power[None]
    elif candidate["blend"] == "geometric":
        gate = agreement * support_power[None]
    elif candidate["blend"] == "support_only":
        gate = support_power[None]
    else:
        raise ValueError("unsupported blend")
    gate = gate / gate.mean(1, keepdim=True).clamp_min(1e-6)
    return gate * (center - text_logits)


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
    candidates = protocol.get("candidate_set", [])
    if len(candidates) != 1:
        raise ValueError("fixed test requires exactly one candidate")
    candidate = candidates[0]
    allowed = {
        ("support_aware_consensus_update", "median", "harmonic_image_support", "inverse", 2.0, 0.25, "linear", 0.75),
        ("support_aware_support_only_update", "median", "min_image_support", "none", 0.0, 0.5, "linear", 0.75),
        ("support_aware_support_only_update", "median", "min_image_support", "none", 0.0, 1.0, "linear", 0.5),
        ("support_aware_support_only_update", "mean", "object_image_support", "none", 0.0, 1.0, "support_only", 1.0),
    }
    signature = (
        candidate.get("name"), candidate.get("aggregation", "median"), candidate.get("support_mode"), candidate.get("consensus"),
        float(candidate.get("tau")), float(candidate.get("support_exponent")),
        candidate.get("blend"), float(candidate.get("strength")),
    )
    if signature not in allowed:
        raise ValueError(f"unexpected fixed candidate: {signature!r}")
    if "do not search" not in protocol.get("test_rule", ""):
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
    val_scores = torch.stack([zscore(item["val"].float()) for item in saved])
    test_scores = torch.stack([zscore(item["test"].float()) for item in saved])
    val_delta = gated_delta(val_scores, val_text, candidate, cache)
    test_delta = gated_delta(test_scores, test_text, candidate, cache)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = float(candidate["strength"])
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
            "individual_test_auc_gain_mean": float(np.mean([row["test_auc_gain"] for row in individual])),
            "individual_test_auc_gain_std": float(np.std([row["test_auc_gain"] for row in individual])),
            "individual_test_hm_gain_mean": float(np.mean([row["test_hm_gain"] for row in individual])),
            "individual_test_hm_gain_std": float(np.std([row["test_hm_gain"] for row in individual])),
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
