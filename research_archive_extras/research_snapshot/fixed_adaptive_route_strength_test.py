"""Protocol-locked test for a validation-selected route-strength candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import re
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
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()
    oc = torch.bincount(po[tp], minlength=len(oi)).float()
    raw = torch.log1p(torch.minimum(ac[pa], oc[po]))
    support = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)
    seen = torch.bincount(tp, minlength=len(meta["pairs"])) > 0
    return support, seen


def entropy_norm(val_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    def ent(x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(x.float(), dim=1)
        return -(p * p.clamp_min(1e-12).log()).sum(1)
    v, x = ent(val_logits), ent(logits)
    return ((x - v.min()) / (v.max() - v.min()).clamp_min(1e-6)).clamp(0, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pp = Path(args.protocol).resolve()
    protocol = json.loads(pp.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol not preregistered")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    probe_path = Path(protocol["validation_probe"]["result"]).resolve()
    if sha256(probe_path) != protocol["validation_probe"]["result_sha256"]:
        raise ValueError("probe SHA mismatch")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("test_read") is not False or probe.get("best") is None:
        raise ValueError("validation probe invalid")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    saved = []
    for item in protocol["sources"]:
        path = Path(item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if set(payload) != {"val", "test"}:
            raise ValueError("fixed sources require val/test")
        saved.append(payload)
    n = int(protocol["family_size"])
    iso_v = torch.stack([zscore(x["val"].float()) for x in saved[:n]]).mean(0)
    diag_v = torch.stack([zscore(x["val"].float()) for x in saved[n:]]).mean(0)
    iso_t = torch.stack([zscore(x["test"].float()) for x in saved[:n]]).mean(0)
    diag_t = torch.stack([zscore(x["test"].float()) for x in saved[n:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(cache["val"]["com"].float() @ text.T)
    test_text = zscore(cache["test"]["com"].float() @ text.T)
    support, seen = support_seen(cache)
    inverse_v = (1 - support).clamp_min(1e-6)[None] * diag_v + support[None] * iso_v
    inverse_t = (1 - support).clamp_min(1e-6)[None] * diag_t + support[None] * iso_t
    soft_v = support.clamp_min(1e-6).pow(0.25)[None] * diag_v + (1 - support.clamp_min(1e-6).pow(0.25))[None] * iso_v
    soft_t = support.clamp_min(1e-6).pow(0.25)[None] * diag_t + (1 - support.clamp_min(1e-6).pow(0.25))[None] * iso_t
    scope = seen.float() + (1 - seen.float()) * 0.25
    a = protocol["candidate"]
    vu = entropy_norm(cache["val"]["logits"], cache["val"]["logits"])
    tu = entropy_norm(cache["val"]["logits"], cache["test"]["logits"])
    if "schedule" in a:
        match = re.fullmatch(r"sigmoid_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)", str(a["schedule"]))
        if match is None:
            raise ValueError(f"unsupported joint schedule: {a['schedule']}")
        slope, threshold = float(match.group(1)), float(match.group(2))
        vu, tu = torch.sigmoid(slope * (vu - threshold)), torch.sigmoid(slope * (tu - threshold))
    else:
        power = float(a["entropy_power"])
        vu, tu = vu.pow(power), tu.pow(power)
    va = seen[cache["val"]["pair"].long()].float() + (1 - seen[cache["val"]["pair"].long()].float()) * vu
    ta = seen[cache["test"]["pair"].long()].float() + (1 - seen[cache["test"]["pair"].long()].float()) * tu
    vp, tp = cache["val"]["pair"].long(), cache["test"]["pair"].long()
    dv = (1-va[:,None]) * (inverse_v-val_text) * float(a["a_strength"]) + va[:,None] * soft_v * scope[vp][:,None] * float(a["b_strength"])
    dt = (1-ta[:,None]) * (inverse_t-test_text) * float(a["a_strength"]) + ta[:,None] * soft_t * scope[tp][:,None] * float(a["b_strength"])
    ve = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    te = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    val, test = cache["val"], cache["test"]
    bv, fv = ve(val["logits"].float(), val["attr"], val["obj"], val["pair"]), ve(val["logits"].float()+dv, val["attr"], val["obj"], val["pair"])
    bt, ft = te(test["logits"].float(), test["attr"], test["obj"], test["pair"]), te(test["logits"].float()+dt, test["attr"], test["obj"], test["pair"])
    result = {"status":"passed","test_read":True,"protocol":str(pp),"protocol_sha256":sha256(pp),"candidate":a,"normalization":"validation_entropy_minmax_reused_for_test","base_val":bv,"fused_val":fv,"val_gain":{"AUC":fv["AUC"]-bv["AUC"],"HM":fv["best_hm"]-bv["best_hm"]},"base_test":bt,"fused_test":ft,"test_gain":{"AUC":ft["AUC"]-bt["AUC"],"HM":ft["best_hm"]-bt["best_hm"]},"sources":[{"path":str(Path(x["path"]).resolve()),"sha256":x["sha256"]} for x in protocol["sources"]],"test_search_after_lock":False}
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"val_gain":result["val_gain"],"test_gain":result["test_gain"]},indent=2))


if __name__ == "__main__":
    main()
