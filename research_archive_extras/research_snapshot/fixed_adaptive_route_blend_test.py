"""Protocol-locked fixed test for an instance-adaptive frozen route blend."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def score(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, route: str, power: float) -> torch.Tensor:
    support, _ = pair_support(cache)
    weight = support.clamp_min(1e-6).pow(power)
    if route == "inverse_pair_support":
        weight = 1.0 - weight
    elif route != "pair_support":
        raise ValueError(route)
    return weight[None] * diag + (1.0 - weight)[None] * iso


def entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)


def val_normalized_entropy(val_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    val_raw = entropy(val_logits)
    raw = entropy(logits)
    return ((raw - val_raw.min()) / (val_raw.max() - val_raw.min()).clamp_min(1e-6)).clamp(0.0, 1.0)


def schedule_alpha(candidate: dict[str, object], uncertainty: torch.Tensor) -> torch.Tensor:
    name = str(candidate["schedule"])
    constant = re.fullmatch(r"constant_([0-9]+(?:\.[0-9]+)?)", name)
    if constant is not None:
        return torch.full_like(uncertainty, float(constant.group(1))).clamp(0.0, 1.0)
    match = re.fullmatch(r"unseen_entropy_([0-9]+(?:\.[0-9]+)?)", name)
    if match is None:
        raise ValueError(f"unsupported schedule: {candidate['schedule']}")
    return uncertainty.pow(float(match.group(1)))


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
    probe_path = Path(protocol["validation_probe"]["result"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"] or sha256(probe_path) != protocol["validation_probe"]["result_sha256"]:
        raise ValueError("cache/probe SHA mismatch")
    validation = json.loads(probe_path.read_text(encoding="utf-8"))
    if validation.get("test_read") is not False:
        raise ValueError("validation probe must be test-free")
    val_source_paths = [Path(item["path"]).resolve() for item in protocol["validation_sources"]]
    for item, path in zip(protocol["validation_sources"], val_source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"validation source SHA mismatch: {path}")
    probe_sources = [Path(item["path"]).resolve() for item in validation["sources"]]
    if [str(p) for p in probe_sources] != [str(p) for p in val_source_paths]:
        raise ValueError("validation probe source list differs from protocol")
    source_paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    family_size = int(protocol["family_size"])
    if any(set(item) != {"val", "test"} for item in saved):
        raise ValueError("fixed-test sources must contain exactly val/test keys")
    iso_val = torch.stack([zscore(item["val"].float()) for item in saved[:family_size]]).mean(0)
    diag_val = torch.stack([zscore(item["val"].float()) for item in saved[family_size:]]).mean(0)
    iso_test = torch.stack([zscore(item["test"].float()) for item in saved[:family_size]]).mean(0)
    diag_test = torch.stack([zscore(item["test"].float()) for item in saved[family_size:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    candidate = protocol["candidate"]
    route_a, route_b = candidate["route_a"], candidate["route_b"]
    score_a_val = score(cache, iso_val, diag_val, route_a["route"], float(route_a["power"]))
    score_a_test = score(cache, iso_test, diag_test, route_a["route"], float(route_a["power"]))
    score_b_val = score(cache, iso_val, diag_val, route_b["route"], float(route_b["power"]))
    score_b_test = score(cache, iso_test, diag_test, route_b["route"], float(route_b["power"]))
    support, pair_seen = pair_support(cache)
    val_pairs = cache["val"]["pair"].long()
    test_pairs = cache["test"]["pair"].long()
    scope_b = pair_seen.float() + (1.0 - pair_seen.float()) * float(route_b["seen_lambda"])
    val_uncertainty = val_normalized_entropy(cache["val"]["logits"], cache["val"]["logits"])
    test_uncertainty = val_normalized_entropy(cache["val"]["logits"], cache["test"]["logits"])
    val_schedule = schedule_alpha(candidate, val_uncertainty)
    test_schedule = schedule_alpha(candidate, test_uncertainty)
    val_alpha = pair_seen[val_pairs].float() + (1.0 - pair_seen[val_pairs].float()) * val_schedule
    test_alpha = pair_seen[test_pairs].float() + (1.0 - pair_seen[test_pairs].float()) * test_schedule
    delta_a_val = score_a_val - val_text
    delta_a_test = score_a_test - test_text
    delta_b_val = score_b_val * scope_b[val_pairs][:, None] * float(route_b["strength"])
    delta_b_test = score_b_test * scope_b[test_pairs][:, None] * float(route_b["strength"])
    val_delta = (1.0 - val_alpha[:, None]) * float(route_a["strength"]) * delta_a_val + val_alpha[:, None] * delta_b_val
    test_delta = (1.0 - test_alpha[:, None]) * float(route_a["strength"]) * delta_a_test + test_alpha[:, None] * delta_b_test
    val, test = cache["val"], cache["test"]
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed",
        "test_read": True,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "candidate": candidate,
        "normalization": "validation_entropy_minmax_reused_for_test",
        "base_val": base_val,
        "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test,
        "fused_test": fused_test,
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
