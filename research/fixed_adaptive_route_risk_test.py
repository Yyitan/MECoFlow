"""Protocol-locked fixed test for entropy-plus-support structural-risk routing."""

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

from dataset import CompositionDataset
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def support_seen(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {name: i for i, name in enumerate(meta["attrs"])}
    oi = {name: i for i, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pa[train_pair], minlength=len(ai)).float()
    oc = torch.bincount(po[train_pair], minlength=len(oi)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    support = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(1)


def normalized_entropy(val_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    v = entropy(val_logits)
    x = entropy(logits)
    return ((x - v.min()) / (v.max() - v.min()).clamp_min(1e-6)).clamp(0.0, 1.0)


def route_score(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, route: str, power: float) -> torch.Tensor:
    support, _ = support_seen(cache)
    weight = support.clamp_min(1e-6).pow(power)
    if route == "inverse_pair_support":
        weight = 1.0 - weight
    elif route != "pair_support":
        raise ValueError(route)
    return weight[None] * diag + (1.0 - weight)[None] * iso


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
    if validation.get("test_read") is not False or validation.get("best") is None:
        raise ValueError("validation probe must be test-free and have a locked candidate")
    val_paths = [Path(item["path"]).resolve() for item in protocol["validation_sources"]]
    for item, path in zip(protocol["validation_sources"], val_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"validation source SHA mismatch: {path}")
    probe_paths = [Path(item["path"]).resolve() for item in validation["sources"]]
    if [str(p) for p in probe_paths] != [str(p) for p in val_paths]:
        raise ValueError("probe source list differs from protocol")
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
    beta, slope, threshold = float(candidate["beta"]), float(candidate["slope"]), float(candidate["threshold"])
    support, pair_seen = support_seen(cache)
    val_pairs = cache["val"]["pair"].long()
    test_pairs = cache["test"]["pair"].long()
    route_a, route_b = candidate["route_a"], candidate["route_b"]
    score_a_val = route_score(cache, iso_val, diag_val, route_a["route"], float(route_a["power"]))
    score_a_test = route_score(cache, iso_test, diag_test, route_a["route"], float(route_a["power"]))
    score_b_val = route_score(cache, iso_val, diag_val, route_b["route"], float(route_b["power"]))
    score_b_test = route_score(cache, iso_test, diag_test, route_b["route"], float(route_b["power"]))
    val_risk = normalized_entropy(cache["val"]["logits"], cache["val"]["logits"]) + beta * (1.0 - support[val_pairs])
    test_risk = normalized_entropy(cache["val"]["logits"], cache["test"]["logits"]) + beta * (1.0 - support[test_pairs])
    # The min/max normalization is fitted on validation risk and reused on test.
    risk_min, risk_max = val_risk.min(), val_risk.max()
    val_risk = ((val_risk - risk_min) / (risk_max - risk_min).clamp_min(1e-6)).clamp(0.0, 1.0)
    test_risk = ((test_risk - risk_min) / (risk_max - risk_min).clamp_min(1e-6)).clamp(0.0, 1.0)
    val_raw = torch.sigmoid(slope * (threshold - val_risk))
    test_raw = torch.sigmoid(slope * (threshold - test_risk))
    val_alpha = pair_seen[val_pairs].float() + (1.0 - pair_seen[val_pairs].float()) * val_raw
    test_alpha = pair_seen[test_pairs].float() + (1.0 - pair_seen[test_pairs].float()) * test_raw
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * float(route_b["seen_lambda"])
    val_delta = (1.0 - val_alpha[:, None]) * float(route_a["strength"]) * (score_a_val - val_text) + val_alpha[:, None] * float(route_b["strength"]) * score_b_val * scope[val_pairs][:, None]
    test_delta = (1.0 - test_alpha[:, None]) * float(route_a["strength"]) * (score_a_test - test_text) + test_alpha[:, None] * float(route_b["strength"]) * score_b_test * scope[test_pairs][:, None]
    val, test = cache["val"], cache["test"]
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed", "test_read": True, "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "candidate": candidate, "normalization": "validation_entropy_plus_support_minmax_reused_for_test",
        "base_val": base_val, "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test, "fused_test": fused_test,
        "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "test_search_after_lock": False,
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain": result["val_gain"], "test_gain": result["test_gain"]}, indent=2))


if __name__ == "__main__":
    main()
