"""Validation-only joint schedule/strength grid for frozen route residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def entropy(x: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(x.float(), dim=1)
    return norm(-(p * p.clamp_min(1e-12).log()).sum(1))


def support_seen(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {n: i for i, n in enumerate(meta["attrs"])}
    oi = {n: i for i, n in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()
    oc = torch.bincount(po[tp], minlength=len(oi)).float()
    s = norm(torch.log1p(torch.minimum(ac[pa], oc[po])))
    seen = torch.bincount(tp, minlength=len(meta["pairs"])) > 0
    return s, seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pp = Path(args.protocol).resolve()
    protocol = json.loads(pp.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_validation":
        raise ValueError("protocol must be preregistered")
    cp = Path(protocol["cache"]).resolve()
    if sha256(cp) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    cache = torch.load(cp, map_location="cpu", weights_only=False)
    val = cache["val"]
    vals = []
    for item in protocol["validation_sources"]:
        path = Path(item["path"]).resolve()
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "val" not in payload:
            raise ValueError("source missing val")
        vals.append(payload["val"].float())
    n = int(protocol["family_size"])
    iso = torch.stack([zscore(x) for x in vals[:n]]).mean(0)
    diag = torch.stack([zscore(x) for x in vals[n:]]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    support, seen = support_seen(cache)
    inverse = (1 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    scope = seen.float() + (1 - seen.float()) * 0.25
    pair = val["pair"].long()
    da = inverse - text_logits
    db = soft * scope[pair][:, None]
    u = entropy(val["logits"])
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    ev = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = ev(val["logits"].float(), val["attr"], val["obj"], pair)
    rows = []
    for slope in (4.0, 8.0, 16.0):
        for threshold in (0.3, 0.5, 0.7):
            raw = torch.sigmoid(slope * (u - threshold))
            alpha = seen[pair].float() + (1 - seen[pair].float()) * raw
            for a_strength in (0.75, 1.0, 1.25):
                for b_strength in (1.0, 1.5, 2.0):
                    fused = val["logits"].float() + (1 - alpha[:, None]) * da * a_strength + alpha[:, None] * db * b_strength
                    stats = ev(fused, val["attr"], val["obj"], pair)
                    rows.append({"slope":slope,"threshold":threshold,"a_strength":a_strength,"b_strength":b_strength,"auc_gain":stats["AUC"]-base["AUC"],"hm_gain":stats["best_hm"]-base["best_hm"],"fused_val":stats})
    eligible = [r for r in rows if r["auc_gain"] >= float(protocol["auc_gate"])]
    result = {"status":"passed","mode":"validation_only_joint_sigmoid_strength","test_read":False,"protocol":str(pp),"protocol_sha256":sha256(pp),"cache":{"path":str(cp),"sha256":sha256(cp)},"candidate_count":len(rows),"selection":"retain AUC gain >= gate and maximize HM, then AUC","base_val":base,"best":max(eligible,key=lambda r:(r["hm_gain"],r["auc_gain"])) if eligible else None,"rows":rows}
    out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({"best":result["best"],"candidate_count":len(rows)},indent=2))


if __name__ == "__main__":
    main()
