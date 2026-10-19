"""Protocol-locked fixed test for candidate-conditioned route blending."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def pair_support(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    raw = torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj]))
    support = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol not preregistered for fixed test")
    cache_path = Path(protocol["cache"]).resolve()
    probe_path = Path(protocol["validation_probe"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"] or sha256(probe_path) != protocol["validation_probe_sha256"]:
        raise ValueError("cache/probe SHA mismatch")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("test_read") is not False:
        raise ValueError("validation probe is not test-free")
    candidate = protocol["candidate"]
    expected_name = f"candidate_score_sigmoid_{float(candidate['slope']):g}_{float(candidate['threshold']):g}"
    if probe["best_balanced"]["name"] != expected_name:
        raise ValueError(f"candidate differs from validation selection: {expected_name}")
    source_paths = [Path(item).resolve() for item in protocol["sources"]]
    for path, expected in zip(source_paths, protocol["source_sha256"]):
        if sha256(path) != expected:
            raise ValueError(f"source SHA mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    if any(set(item) != {"val", "test"} for item in saved):
        raise ValueError("fixed sources must contain exactly val/test keys")
    family_size = len(saved) // 2
    iso_val = torch.stack([zscore(item["val"].float()) for item in saved[:family_size]]).mean(0)
    diag_val = torch.stack([zscore(item["val"].float()) for item in saved[family_size:]]).mean(0)
    iso_test = torch.stack([zscore(item["test"].float()) for item in saved[:family_size]]).mean(0)
    diag_test = torch.stack([zscore(item["test"].float()) for item in saved[family_size:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    support, pair_seen = pair_support(cache)
    inverse_val = (1.0 - support)[None] * diag_val + support[None] * iso_val
    inverse_test = (1.0 - support)[None] * diag_test + support[None] * iso_test
    soft_weight = support.clamp_min(1e-6).pow(0.25)
    soft_val = soft_weight[None] * diag_val + (1.0 - soft_weight)[None] * iso_val
    soft_test = soft_weight[None] * diag_test + (1.0 - soft_weight)[None] * iso_test
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    val_pairs, test_pairs = cache["val"]["pair"].long(), cache["test"]["pair"].long()
    val_pair_score, test_pair_score = zscore(cache["val"]["logits"].float()), zscore(cache["test"]["logits"].float())
    val_alpha = torch.sigmoid(float(candidate["slope"]) * (val_pair_score - float(candidate["threshold"])))
    test_alpha = torch.sigmoid(float(candidate["slope"]) * (test_pair_score - float(candidate["threshold"])))
    val_alpha[:, pair_seen] = 1.0
    test_alpha[:, pair_seen] = 1.0
    val_delta_a, test_delta_a = inverse_val - val_text, inverse_test - test_text
    val_delta_b = soft_val * scope[val_pairs][:, None] * 1.5
    test_delta_b = soft_test * scope[test_pairs][:, None] * 1.5
    val_delta = (1.0 - val_alpha) * val_delta_a + val_alpha * val_delta_b
    test_delta = (1.0 - test_alpha) * test_delta_a + test_alpha * test_delta_b
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    val, test = cache["val"], cache["test"]
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed", "test_read": True,
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "candidate": candidate,
        "base_val": base_val, "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test, "fused_test": fused_test,
        "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "test_search_after_lock": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain": result["val_gain"], "test_gain": result["test_gain"]}, indent=2))


if __name__ == "__main__":
    main()
