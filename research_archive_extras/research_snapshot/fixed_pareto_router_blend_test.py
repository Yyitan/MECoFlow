"""Protocol-locked fixed test for a blend of two frozen routing candidates."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator, real_centroids, zscore  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    raw = torch.log1p(torch.minimum(ac[pair_attr], oc[pair_obj]))
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


def score(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, route: str, power: float) -> torch.Tensor:
    support = pair_support(cache)
    weight = support.clamp_min(1e-6).pow(power)
    if route == "inverse_pair_support":
        weight = 1.0 - weight
    elif route != "pair_support":
        raise ValueError(route)
    return weight[None] * diag + (1.0 - weight[None]) * iso


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
    source_paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], source_paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    n = int(protocol["family_size"])
    iso_val = torch.stack([zscore(item["val"].float()) for item in saved[:n]]).mean(0)
    diag_val = torch.stack([zscore(item["val"].float()) for item in saved[n:]]).mean(0)
    iso_test = torch.stack([zscore(item["test"].float()) for item in saved[:n]]).mean(0)
    diag_test = torch.stack([zscore(item["test"].float()) for item in saved[n:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    candidate = protocol["candidate"]
    route_a = candidate["route_a"]
    route_b = candidate["route_b"]
    score_a_val = score(cache, iso_val, diag_val, route_a["route"], float(route_a["power"]))
    score_a_test = score(cache, iso_test, diag_test, route_a["route"], float(route_a["power"]))
    score_b_val = score(cache, iso_val, diag_val, route_b["route"], float(route_b["power"]))
    score_b_test = score(cache, iso_test, diag_test, route_b["route"], float(route_b["power"]))
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    scope_b = unseen + (1.0 - unseen) * float(route_b["seen_lambda"])
    delta_a_val = score_a_val - val_text if route_a["delta"] == "update" else score_a_val
    delta_a_test = score_a_test - test_text if route_a["delta"] == "update" else score_a_test
    delta_b_val = score_b_val * scope_b[None] - val_text if route_b["delta"] == "update" else score_b_val * scope_b[None]
    delta_b_test = score_b_test * scope_b[None] - test_text if route_b["delta"] == "update" else score_b_test * scope_b[None]
    blend = float(candidate["blend_soft"])
    val_delta = (1.0 - blend) * float(route_a["strength"]) * delta_a_val + blend * float(route_b["strength"]) * delta_b_val
    test_delta = (1.0 - blend) * float(route_a["strength"]) * delta_a_test + blend * float(route_b["strength"]) * delta_b_test
    val, test = cache["val"], cache["test"]
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    fused_val = val_eval(val["logits"].float() + val_delta, val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_test = test_eval(test["logits"].float() + test_delta, test["attr"], test["obj"], test["pair"])
    result = {"status":"passed","test_read":True,"protocol":str(protocol_path),"protocol_sha256":sha256(protocol_path),"candidate":candidate,"base_val":base_val,"fused_val":fused_val,"val_gain":{"AUC":fused_val["AUC"]-base_val["AUC"],"HM":fused_val["best_hm"]-base_val["best_hm"]},"base_test":base_test,"fused_test":fused_test,"test_gain":{"AUC":fused_test["AUC"]-base_test["AUC"],"HM":fused_test["best_hm"]-base_test["best_hm"]},"sources":[{"path":str(path),"sha256":sha256(path)} for path in source_paths],"test_search_after_lock":False}
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain":result["val_gain"],"test_gain":result["test_gain"]}, indent=2))


if __name__ == "__main__":
    main()
