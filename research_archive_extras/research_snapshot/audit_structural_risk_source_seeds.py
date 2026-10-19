"""Audit a frozen structural-risk gate across explicitly listed source pairs."""

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
    ai = {n: i for i, n in enumerate(meta["attrs"])}
    oi = {n: i for i, n in enumerate(meta["objs"])}
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


def risk_norm(val_logits: torch.Tensor, logits: torch.Tensor, support: torch.Tensor, pair: torch.Tensor, beta: float, fit_min: torch.Tensor | None = None, fit_max: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = entropy(logits) + beta * (1.0 - support[pair.long()])
    if fit_min is None:
        fit_min, fit_max = raw.min(), raw.max()
    return ((raw - fit_min) / (fit_max - fit_min).clamp_min(1e-6)).clamp(0.0, 1.0), fit_min, fit_max


def route(cache: dict[str, object], iso: torch.Tensor, other: torch.Tensor, name: str, power: float) -> torch.Tensor:
    support, _ = support_seen(cache)
    weight = support.clamp_min(1e-6).pow(power)
    if name == "inverse_pair_support":
        weight = 1.0 - weight
    elif name != "pair_support":
        raise ValueError(name)
    return weight[None] * other + (1.0 - weight)[None] * iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    support, pair_seen = support_seen(cache)
    pair_v, pair_t = val["pair"].long(), test["pair"].long()
    candidate = protocol["candidate"]
    beta, slope, threshold = float(candidate["beta"]), float(candidate["slope"]), float(candidate["threshold"])
    val_risk, risk_min, risk_max = risk_norm(val["logits"], val["logits"], support, pair_v, beta)
    test_risk, _, _ = risk_norm(val["logits"], test["logits"], support, pair_t, beta, risk_min, risk_max)
    val_raw = torch.sigmoid(slope * (threshold - val_risk))
    test_raw = torch.sigmoid(slope * (threshold - test_risk))
    val_alpha = pair_seen[pair_v].float() + (1.0 - pair_seen[pair_v].float()) * val_raw
    test_alpha = pair_seen[pair_t].float() + (1.0 - pair_seen[pair_t].float()) * test_raw
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * float(candidate["route_b"]["seen_lambda"])
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], pair_v)
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], pair_t)
    rows = []
    for index, item in enumerate(protocol["source_pairs"]):
        iso_path, other_path = Path(item["route_a_source"]).resolve(), Path(item["route_b_source"]).resolve()
        if sha256(iso_path) != item["route_a_sha256"] or sha256(other_path) != item["route_b_sha256"]:
            raise ValueError(f"source SHA mismatch at run {index}")
        iso, other = torch.load(iso_path, map_location="cpu", weights_only=True), torch.load(other_path, map_location="cpu", weights_only=True)
        if set(iso) != {"val", "test"} or set(other) != {"val", "test"}:
            raise ValueError("source pairs must contain val/test")
        a_v = route(cache, zscore(iso["val"].float()), zscore(other["val"].float()), candidate["route_a"]["route"], float(candidate["route_a"]["power"]))
        a_t = route(cache, zscore(iso["test"].float()), zscore(other["test"].float()), candidate["route_a"]["route"], float(candidate["route_a"]["power"]))
        b_v = route(cache, zscore(iso["val"].float()), zscore(other["val"].float()), candidate["route_b"]["route"], float(candidate["route_b"]["power"]))
        b_t = route(cache, zscore(iso["test"].float()), zscore(other["test"].float()), candidate["route_b"]["route"], float(candidate["route_b"]["power"]))
        dv = (1.0 - val_alpha[:, None]) * (a_v - val_text) + val_alpha[:, None] * b_v * scope[pair_v][:, None] * float(candidate["route_b"]["strength"])
        dt = (1.0 - test_alpha[:, None]) * (a_t - test_text) + test_alpha[:, None] * b_t * scope[pair_t][:, None] * float(candidate["route_b"]["strength"])
        fused_val = val_eval(val["logits"].float() + dv, val["attr"], val["obj"], pair_v)
        fused_test = test_eval(test["logits"].float() + dt, test["attr"], test["obj"], pair_t)
        rows.append({"run": item["name"], "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]}, "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]}, "route_a_source": str(iso_path), "route_b_source": str(other_path)})
    result = {"status":"passed","test_read":True,"protocol":str(protocol_path),"protocol_sha256":sha256(protocol_path),"candidate":candidate,"test_search_after_lock":False,"runs":rows,"positive_test_auc_runs":sum(x["test_gain"]["AUC"] > 0 for x in rows),"positive_test_hm_runs":sum(x["test_gain"]["HM"] > 0 for x in rows),"mean_test_auc_gain":sum(x["test_gain"]["AUC"] for x in rows)/len(rows),"mean_test_hm_gain":sum(x["test_gain"]["HM"] for x in rows)/len(rows)}
    out = Path(args.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
