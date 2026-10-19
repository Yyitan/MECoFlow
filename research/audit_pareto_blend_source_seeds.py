"""Source-seed audit for a frozen Pareto route blend."""

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


def support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]
    ai = {name: index for index, name in enumerate(meta["attrs"])}
    oi = {name: index for index, name in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()
    oc = torch.bincount(po[tp], minlength=len(oi)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


def route(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, name: str, power: float) -> torch.Tensor:
    weight = support(cache).clamp_min(1e-6).pow(power)
    if name == "inverse_pair_support":
        weight = 1.0 - weight
    elif name != "pair_support":
        raise ValueError(name)
    return weight[None] * diag + (1.0 - weight[None]) * iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pp = Path(args.protocol).resolve()
    p = json.loads(pp.read_text(encoding="utf-8"))
    cp = Path(p["cache"]).resolve()
    cache = torch.load(cp, map_location="cpu", weights_only=False)
    paths = [Path(item["path"]).resolve() for item in p["sources"]]
    for item, path in zip(p["sources"], paths):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
    n = int(p["family_size"])
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    val, test = cache["val"], cache["test"]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    _, counts = real_centroids(cache)
    unseen = (counts == 0).float()
    soft_scope = unseen + (1.0 - unseen) * float(p["candidate"]["route_b"]["seen_lambda"])
    a, b = p["candidate"]["route_a"], p["candidate"]["route_b"]
    ve = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    te = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    rows = []
    for index in range(n):
        iso_v, diag_v = zscore(saved[index]["val"].float()), zscore(saved[n + index]["val"].float())
        iso_t, diag_t = zscore(saved[index]["test"].float()), zscore(saved[n + index]["test"].float())
        av, at = route(cache, iso_v, diag_v, a["route"], float(a["power"])), route(cache, iso_t, diag_t, a["route"], float(a["power"]))
        bv, bt = route(cache, iso_v, diag_v, b["route"], float(b["power"])), route(cache, iso_t, diag_t, b["route"], float(b["power"]))
        da_v = av - val_text if a["delta"] == "update" else av
        da_t = at - test_text if a["delta"] == "update" else at
        db_v = bv * soft_scope[None] - val_text if b["delta"] == "update" else bv * soft_scope[None]
        db_t = bt * soft_scope[None] - test_text if b["delta"] == "update" else bt * soft_scope[None]
        blend = float(p["candidate"]["blend_soft"])
        dv = (1.0 - blend) * float(a["strength"]) * da_v + blend * float(b["strength"]) * db_v
        dt = (1.0 - blend) * float(a["strength"]) * da_t + blend * float(b["strength"]) * db_t
        base_v = ve(val["logits"].float(), val["attr"], val["obj"], val["pair"])
        fused_v = ve(val["logits"].float() + dv, val["attr"], val["obj"], val["pair"])
        base_t = te(test["logits"].float(), test["attr"], test["obj"], test["pair"])
        fused_t = te(test["logits"].float() + dt, test["attr"], test["obj"], test["pair"])
        rows.append({"seed_index": index, "val_gain":{"AUC":fused_v["AUC"]-base_v["AUC"],"HM":fused_v["best_hm"]-base_v["best_hm"]},"test_gain":{"AUC":fused_t["AUC"]-base_t["AUC"],"HM":fused_t["best_hm"]-base_t["best_hm"]}})
    result = {"status":"passed","protocol":str(pp),"protocol_sha256":sha256(pp),"candidate":p["candidate"],"test_search_after_lock":False,"runs":rows,"positive_test_auc_runs":sum(row["test_gain"]["AUC"]>0 for row in rows),"positive_test_hm_runs":sum(row["test_gain"]["HM"]>0 for row in rows)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
