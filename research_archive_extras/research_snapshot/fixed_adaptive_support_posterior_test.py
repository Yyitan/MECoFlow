"""Protocol-locked fixed test for graph-conditioned adaptive source posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

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


def row_norm(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def confidence(scores: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(scores, 2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    prob = torch.softmax(scores, dim=2)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(2)
    return row_norm(margin) if name == "margin_high" else 1.0 - row_norm(entropy)


def source_posterior(scores: torch.Tensor, c: dict[str, object]) -> torch.Tensor:
    center = scores.median(0).values if c["center"] == "median" else scores.mean(0)
    residual = scores - center[None]
    d = residual.abs().mean(2) if c["distance"] == "abs" else residual.square().mean(2).sqrt()
    d = d / d.mean(1, keepdim=True).clamp_min(1e-6)
    w = torch.exp(-float(c["tau"]) * d) * confidence(scores, c["confidence"]).clamp_min(1e-6).pow(float(c["confidence_power"]))
    w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
    return (w[:, :, None] * scores).sum(0)


def support(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    ai = {v: i for i, v in enumerate(meta["attrs"])}
    oi = {v: i for i, v in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()[pa]
    oc = torch.bincount(po[tp], minlength=len(oi)).float()[po]
    raw = {"min": torch.minimum(ac, oc), "harmonic": 2 * ac * oc / (ac + oc).clamp_min(1e-6), "attribute": ac, "object": oc}[mode]
    raw = torch.log1p(raw)
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


def resolve(value: str, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered")
    candidates = protocol.get("candidate_set", [])
    if len(candidates) != 1:
        raise ValueError("exactly one candidate required")
    c = candidates[0]
    required = {"center", "distance", "confidence", "tau", "confidence_power", "delta", "support", "support_power", "blend", "strength"}
    if set(c) != required:
        raise ValueError(f"candidate keys mismatch: {sorted(c)}")
    if "validation AUC gain" not in protocol.get("validation_selection_rule", "") or "largest validation HM" not in protocol.get("validation_selection_rule", ""):
        raise ValueError("validation selection rule incomplete")
    if "do not search" not in protocol.get("test_rule", "") or protocol.get("validation_gate", {}).get("test_read") is not False:
        raise ValueError("protocol test rule/gate invalid")
    cache_path = resolve(protocol["cache"], root)
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache sha mismatch")
    source_paths = [resolve(x["path"], root) for x in protocol["sources"]]
    for path, item in zip(source_paths, protocol["sources"]):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source sha mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(x, map_location="cpu", weights_only=True) for x in source_paths]
    val_scores = torch.stack([zscore(x["val"].float()) for x in saved])
    test_scores = torch.stack([zscore(x["test"].float()) for x in saved])
    val_source = source_posterior(val_scores, c)
    test_source = source_posterior(test_scores, c)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text, test_text = zscore(val["com"].float() @ text.T), zscore(test["com"].float() @ text.T)
    val_delta = val_source if c["delta"] == "full" else val_source - val_text
    test_delta = test_source if c["delta"] == "full" else test_source - test_text
    s = support(cache, c["support"]).clamp_min(1e-6).pow(float(c["support_power"]))
    val_gate = s[None] if c["blend"] == "product" else 0.5 + 0.5 * s[None]
    val_gate = val_gate / val_gate.mean().clamp_min(1e-6)
    test_gate = val_gate
    ev = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    et = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = ev(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = et(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    fused_val = ev(val["logits"].float() + float(c["strength"]) * val_gate * val_delta, val["attr"], val["obj"], val["pair"])
    fused_test = et(test["logits"].float() + float(c["strength"]) * test_gate * test_delta, test["attr"], test["obj"], test["pair"])
    val_auc, val_hm = fused_val["AUC"] - base_val["AUC"], fused_val["best_hm"] - base_val["best_hm"]
    expected = protocol["validation_gate"]
    if abs(val_auc - float(expected["mean_auc_gain"])) > 1e-10 or abs(val_hm - float(expected["mean_hm_gain"])) > 1e-10:
        raise ValueError(f"validation replay mismatch {(val_auc, val_hm)}")
    result = {"status": "passed", "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path), "candidate": c, "cache": str(cache_path), "cache_sha256": sha256(cache_path), "sources": [{"path": str(x), "sha256": sha256(x)} for x in source_paths], "selection_locked": True, "test_read": True}, "base_val": base_val, "fused_val": fused_val, "base_test": base_test, "fused_test": fused_test, "mean": {"val_auc_gain": val_auc, "val_hm_gain": val_hm, "test_auc_gain": fused_test["AUC"] - base_test["AUC"], "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"]}, "source_count": len(source_paths), "test_search_after_lock": False}
    out = resolve(args.output, root); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
