"""Audit per-generator-seed outcomes for a frozen selective router."""

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


def route_weight(cache: dict[str, object], route: str, power: float) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    if route == "pair_support":
        raw = torch.log1p(torch.minimum(ac[pair_attr], oc[pair_obj]))
    elif route == "obj_support":
        raw = torch.log1p(oc[pair_obj])
    else:
        raise ValueError(route)
    weight = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    return weight.clamp_min(1e-6).pow(power)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache_path = Path(protocol["cache"]).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    sources = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], sources):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    n = int(protocol["family_size"])
    candidate = protocol["candidate"]
    weight = route_weight(cache, candidate["route"], float(candidate["power"]))
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    if candidate["scope"] == "unseen_only":
        scope = unseen
    elif candidate["scope"] == "seen_only":
        scope = 1.0 - unseen
    elif candidate["scope"] == "all":
        scope = torch.ones_like(unseen)
    elif candidate["scope"] == "soft_tapered":
        seen_lambda = float(candidate["seen_lambda"])
        if not 0.0 <= seen_lambda <= 1.0:
            raise ValueError("seen_lambda must be in [0, 1]")
        scope = unseen + (1.0 - unseen) * seen_lambda
    else:
        raise ValueError(candidate["scope"])
    val = cache["val"]
    test = cache["test"]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    runs = []
    for index in range(n):
        iso = torch.load(sources[index], map_location="cpu", weights_only=True)
        diag = torch.load(sources[n + index], map_location="cpu", weights_only=True)
        iso_val, diag_val = zscore(iso["val"].float()), zscore(diag["val"].float())
        iso_test, diag_test = zscore(iso["test"].float()), zscore(diag["test"].float())
        val_score = weight[None] * diag_val + (1.0 - weight[None]) * iso_val
        test_score = weight[None] * diag_test + (1.0 - weight[None]) * iso_test
        val_delta = val_score if candidate["delta"] == "full" else val_score - val_text
        test_delta = test_score if candidate["delta"] == "full" else test_score - test_text
        strength = float(candidate["strength"])
        base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
        fused_val = val_eval(val["logits"].float() + strength * val_delta * scope[None], val["attr"], val["obj"], val["pair"])
        base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
        fused_test = test_eval(test["logits"].float() + strength * test_delta * scope[None], test["attr"], test["obj"], test["pair"])
        runs.append({"seed_index": index, "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]}, "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]}})
    result = {"status": "passed", "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path), "candidate": candidate, "test_search_after_lock": False, "runs": runs, "positive_test_auc_runs": sum(run["test_gain"]["AUC"] > 0 for run in runs), "positive_test_hm_runs": sum(run["test_gain"]["HM"] > 0 for run in runs)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
