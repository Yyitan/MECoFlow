"""Protocol-locked fixed test for train-support pair-level source routing."""

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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def support(cache: dict[str, object]) -> torch.Tensor:
    meta = cache["meta"]; attrs = {n: i for i, n in enumerate(meta["attrs"])}; objs = {n: i for i, n in enumerate(meta["objs"])}
    pa = torch.tensor([attrs[a] for a, _ in meta["pairs"]]); po = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long(); ac = torch.bincount(pa[train_pair], minlength=len(attrs)).float(); oc = torch.bincount(po[train_pair], minlength=len(objs)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po])); return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


def routed(cache: dict[str, object], iso: torch.Tensor, diag: torch.Tensor, route: str, power: float) -> torch.Tensor:
    if route in {"disagreement_reliability", "inverse_disagreement_reliability"}:
        iso_mean, diag_mean = iso.mean(0), diag.mean(0)
        iso_disp = (iso - iso_mean).abs().mean(0).mean(0)
        diag_disp = (diag - diag_mean).abs().mean(0).mean(0)
        w = (1.0 / (diag_disp + 1e-6)) / (1.0 / (diag_disp + 1e-6) + 1.0 / (iso_disp + 1e-6))
        lo, hi = w.min(), w.max(); w = (w - lo) / (hi - lo).clamp_min(1e-6)
        if route == "inverse_disagreement_reliability":
            w = 1.0 - w
    else:
        w = support(cache).clamp_min(1e-6)
        iso_mean, diag_mean = (iso.mean(0), diag.mean(0)) if iso.dim() == 3 else (iso, diag)
    if route in {"disagreement_reliability", "inverse_disagreement_reliability"}:
        pass
    elif route == "inverse_pair_support":
        w = 1.0 - w
    elif route != "pair_support":
        raise ValueError(route)
    w = w.clamp_min(1e-6).pow(power)
    return w[None, :] * diag_mean + (1.0 - w[None, :]) * iso_mean


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True); parser.add_argument("--dataset-path", required=True); parser.add_argument("--output", required=True); args = parser.parse_args()
    pp = Path(args.protocol).resolve(); protocol = json.loads(pp.read_text(encoding="utf-8"));
    if protocol.get("status") != "preregistered_before_fixed_test": raise ValueError("protocol not preregistered")
    cp = Path(protocol["cache"]).resolve(); vp = Path(protocol["validation_probe"]["result"]).resolve()
    if sha256(cp) != protocol["cache_sha256"] or sha256(vp) != protocol["validation_probe"]["result_sha256"]: raise ValueError("cache/probe SHA mismatch")
    validation = json.loads(vp.read_text(encoding="utf-8"));
    if validation.get("test_read") is not False: raise ValueError("validation probe must be test-free")
    paths = [Path(item["path"]).resolve() for item in protocol["sources"]]
    for item, path in zip(protocol["sources"], paths):
        if sha256(path) != item["sha256"]: raise ValueError(f"source SHA mismatch: {path}")
    cache = torch.load(cp, map_location="cpu", weights_only=False); val = cache["val"]; test = cache["test"]
    saved = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]; n = int(protocol["family_size"])
    iso_val = torch.stack([zscore(x["val"].float()) for x in saved[:n]]); diag_val = torch.stack([zscore(x["val"].float()) for x in saved[n:]])
    iso_test = torch.stack([zscore(x["test"].float()) for x in saved[:n]]); diag_test = torch.stack([zscore(x["test"].float()) for x in saved[n:]])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1); val_text = zscore(val["com"].float() @ text.T); test_text = zscore(test["com"].float() @ text.T)
    power = float(protocol["candidate"]["power"]); strength = float(protocol["candidate"]["strength"]); delta_name = protocol["candidate"]["delta"]
    route = protocol["candidate"]["route"]
    val_score = routed(cache, iso_val, diag_val, route, power); test_score = routed(cache, iso_test, diag_test, route, power)
    val_delta = val_score if delta_name == "full" else val_score - val_text; test_delta = test_score if delta_name == "full" else test_score - test_text
    ev = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")); et = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = ev(val["logits"].float(), val["attr"], val["obj"], val["pair"]); fused_val = ev(val["logits"].float()+strength*val_delta, val["attr"], val["obj"], val["pair"])
    base_test = et(test["logits"].float(), test["attr"], test["obj"], test["pair"]); fused_test = et(test["logits"].float()+strength*test_delta, test["attr"], test["obj"], test["pair"])
    result = {"status":"passed","test_read":True,"protocol":str(pp),"protocol_sha256":sha256(pp),"candidate":protocol["candidate"],"base_val":base_val,"fused_val":fused_val,"val_gain":{"AUC":fused_val["AUC"]-base_val["AUC"],"HM":fused_val["best_hm"]-base_val["best_hm"]},"base_test":base_test,"fused_test":fused_test,"test_gain":{"AUC":fused_test["AUC"]-base_test["AUC"],"HM":fused_test["best_hm"]-base_test["best_hm"]},"sources":[{"path":str(p),"sha256":sha256(p)} for p in paths],"test_search_after_lock":False}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"val_gain":result["val_gain"],"test_gain":result["test_gain"]},indent=2))


if __name__ == "__main__": main()
