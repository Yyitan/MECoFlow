"""Protocol-locked fixed test for an exploratory AUC-first pair router."""

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


def route_weight(cache: dict[str, object], route: str, power: float) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    if route == "obj_support":
        raw = torch.log1p(obj_count[pair_obj])
    elif route == "attr_support":
        raw = torch.log1p(attr_count[pair_attr])
    elif route == "pair_support":
        raw = torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj]))
    elif route == "inverse_pair_support":
        raw = torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj]))
        raw = 1.0 - (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
        return raw.clamp_min(1e-6).pow(power)
    else:
        raise ValueError(f"unsupported route: {route}")
    raw = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    return raw.clamp_min(1e-6).pow(power)


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
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    validation = json.loads(probe_path.read_text(encoding="utf-8"))
    if sha256(probe_path) != protocol["validation_probe"]["result_sha256"]:
        raise ValueError("validation probe SHA mismatch")
    if validation.get("test_read") is not False:
        raise ValueError("validation probe must be test-free")

    source_paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    family_size = int(protocol["family_size"])
    iso_val = torch.stack([zscore(item["val"].float()) for item in saved[:family_size]])
    diag_val = torch.stack([zscore(item["val"].float()) for item in saved[family_size:]])
    iso_test = torch.stack([zscore(item["test"].float()) for item in saved[:family_size]])
    diag_test = torch.stack([zscore(item["test"].float()) for item in saved[family_size:]])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    candidate = protocol["candidate"]
    weight = route_weight(cache, candidate["route"], float(candidate["power"]))
    val_score = weight[None] * diag_val.mean(0) + (1.0 - weight[None]) * iso_val.mean(0)
    test_score = weight[None] * diag_test.mean(0) + (1.0 - weight[None]) * iso_test.mean(0)
    val_delta = val_score if candidate["delta"] == "full" else val_score - val_text
    test_delta = test_score if candidate["delta"] == "full" else test_score - test_text

    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    val = cache["val"]
    test = cache["test"]
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + float(candidate["strength"]) * val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + float(candidate["strength"]) * test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed",
        "test_read": True,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "candidate": candidate,
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
