"""Validation-only strength calibration for two frozen route residuals."""

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


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), dim=1)
    return normalize(-(p * p.clamp_min(1e-12).log()).sum(1))


def pair_support(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    ai = {n: i for i, n in enumerate(meta["attrs"])}
    oi = {n: i for i, n in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pa[train_pair], minlength=len(ai)).float()
    oc = torch.bincount(po[train_pair], minlength=len(oi)).float()
    support = normalize(torch.log1p(torch.minimum(ac[pa], oc[po])))
    seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return support, seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    pp = Path(args.protocol).resolve()
    protocol = json.loads(pp.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_validation":
        raise ValueError("protocol must be preregistered before validation")
    cache_path = Path(protocol["cache"]).resolve()
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache SHA mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
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
    support, pair_seen = pair_support(cache)
    inverse = (1.0 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    delta_a = inverse - text_logits
    delta_b = soft * scope[val["pair"].long()][:, None]
    u = entropy(val["logits"])
    alpha = pair_seen[val["pair"].long()].float() + (1.0 - pair_seen[val["pair"].long()].float()) * u
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator
    ev = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = ev(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for a_strength in (0.5, 0.75, 1.0, 1.25, 1.5):
        for b_strength in (1.0, 1.25, 1.5, 1.75, 2.0):
            for power in (0.5, 1.0, 2.0):
                schedule = pair_seen[val["pair"].long()].float() + (1.0 - pair_seen[val["pair"].long()].float()) * u.pow(power)
                fused = val["logits"].float() + (1.0 - schedule[:, None]) * delta_a * a_strength + schedule[:, None] * delta_b * b_strength
                stats = ev(fused, val["attr"], val["obj"], val["pair"])
                rows.append({"a_strength": a_strength, "b_strength": b_strength, "entropy_power": power, "auc_gain": stats["AUC"] - base["AUC"], "hm_gain": stats["best_hm"] - base["best_hm"], "fused_val": stats})
    eligible = [r for r in rows if r["auc_gain"] >= float(protocol["auc_gate"])]
    result = {"status":"passed", "mode":"validation_only_route_strength_grid", "test_read":False, "protocol":str(pp), "protocol_sha256":sha256(pp), "cache":{"path":str(cache_path),"sha256":sha256(cache_path)}, "base_val":base, "selection":"retain AUC gain >= gate and maximize HM, then AUC", "candidate_count":len(rows), "best":max(eligible,key=lambda r:(r["hm_gain"],r["auc_gain"])) if eligible else None, "rows":rows}
    out = Path(args.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"best":result["best"],"candidate_count":len(rows)},indent=2))


if __name__ == "__main__":
    main()
