"""Source-pair audit for a locked candidate-conditioned route."""

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
from research.fast_feature_evaluation import FastClosedWorldEvaluator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def pair_support(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pa = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    po = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pa[train_pair], minlength=len(attrs)).float()
    oc = torch.bincount(po[train_pair], minlength=len(objs)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    support = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def evaluate(cache, iso, diag, slope, threshold, dataset_path):
    support, seen = pair_support(cache)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val, test = cache["val"], cache["test"]
    val_text, test_text = zscore(val["com"].float() @ text.T), zscore(test["com"].float() @ text.T)
    inverse_val = (1 - support)[None] * diag["val"] + support[None] * iso["val"]
    inverse_test = (1 - support)[None] * diag["test"] + support[None] * iso["test"]
    sw = support.clamp_min(1e-6).pow(0.25)
    soft_val = sw[None] * diag["val"] + (1 - sw)[None] * iso["val"]
    soft_test = sw[None] * diag["test"] + (1 - sw)[None] * iso["test"]
    scope = seen.float() + (1 - seen.float()) * 0.25
    vp, tp = val["pair"].long(), test["pair"].long()
    va = torch.sigmoid(slope * (zscore(val["logits"].float()) - threshold))
    ta = torch.sigmoid(slope * (zscore(test["logits"].float()) - threshold))
    va[:, seen] = 1.0; ta[:, seen] = 1.0
    vd = (1-va) * (inverse_val-val_text) + va * soft_val * scope[vp][:,None] * 1.5
    td = (1-ta) * (inverse_test-test_text) + ta * soft_test * scope[tp][:,None] * 1.5
    ve = FastClosedWorldEvaluator(CompositionDataset(dataset_path, phase="val"), torch.device("cpu"))
    te = FastClosedWorldEvaluator(CompositionDataset(dataset_path, phase="test"), torch.device("cpu"))
    bv, fv = ve(val["logits"].float(), val["attr"], val["obj"], val["pair"]), ve(val["logits"].float()+vd, val["attr"], val["obj"], val["pair"])
    bt, ft = te(test["logits"].float(), test["attr"], test["obj"], test["pair"]), te(test["logits"].float()+td, test["attr"], test["obj"], test["pair"])
    return {"val_auc_gain": fv["AUC"]-bv["AUC"], "val_hm_gain": fv["best_hm"]-bv["best_hm"], "test_auc_gain": ft["AUC"]-bt["AUC"], "test_hm_gain": ft["best_hm"]-bt["best_hm"]}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--protocol", required=True); p.add_argument("--dataset-path", required=True); p.add_argument("--output", required=True)
    args = p.parse_args(); pp = Path(args.protocol).resolve(); protocol = json.loads(pp.read_text())
    if protocol.get("status") != "preregistered_before_fixed_test": raise ValueError("protocol status")
    cache_path = Path(protocol["cache"]).resolve(); probe_path = Path(protocol["validation_probe"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"] or sha256(probe_path) != protocol["validation_probe_sha256"]: raise ValueError("cache/probe SHA")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    sources = [Path(x).resolve() for x in protocol["sources"]]
    for path, expected in zip(sources, protocol["source_sha256"]):
        if sha256(path) != expected: raise ValueError(f"source SHA {path}")
    saved = [torch.load(x, map_location="cpu", weights_only=True) for x in sources]
    if any(set(x) != {"val", "test"} for x in saved): raise ValueError("source keys")
    out = []
    for i in range(3):
        iso = {k: zscore(saved[i][k].float()) for k in ("val", "test")}
        diag = {k: zscore(saved[i+3][k].float()) for k in ("val", "test")}
        row = evaluate(cache, iso, diag, float(protocol["candidate"]["slope"]), float(protocol["candidate"]["threshold"]), args.dataset_path)
        row["seed"] = i; out.append(row)
    result = {"status":"passed", "test_read":True, "protocol":str(pp), "protocol_sha256":sha256(pp), "candidate":protocol["candidate"], "runs":out, "mean_test_auc_gain":sum(x["test_auc_gain"] for x in out)/3, "mean_test_hm_gain":sum(x["test_hm_gain"] for x in out)/3, "all_test_auc_positive":all(x["test_auc_gain"]>0 for x in out), "all_test_hm_positive":all(x["test_hm_gain"]>0 for x in out), "test_search_after_lock":False}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
