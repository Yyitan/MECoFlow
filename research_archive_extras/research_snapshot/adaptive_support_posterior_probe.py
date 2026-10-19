"""Validation-only graph-conditioned adaptive source-posterior probe."""

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


def row_norm(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-6)


def confidence(scores: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(scores, 2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    prob = torch.softmax(scores, dim=2)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(2)
    if name == "margin_high":
        return row_norm(margin)
    if name == "entropy_low":
        return 1.0 - row_norm(entropy)
    raise ValueError(name)


def source_posterior(scores: torch.Tensor, center: str, distance: str, tau: float, conf_name: str, conf_power: float) -> torch.Tensor:
    c = scores.median(0).values if center == "median" else scores.mean(0)
    residual = scores - c[None]
    d = residual.abs().mean(2) if distance == "abs" else residual.square().mean(2).sqrt()
    d = d / d.mean(1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-tau * d) * confidence(scores, conf_name).clamp_min(1e-6).pow(conf_power)
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    return (weights[:, :, None] * scores).sum(0)


def support_gate(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {v: i for i, v in enumerate(meta["attrs"])}
    objs = {v: i for i, v in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    ac = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()[pair_attr]
    oc = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()[pair_obj]
    if mode == "min": raw = torch.minimum(ac, oc)
    elif mode == "harmonic": raw = 2 * ac * oc / (ac + oc).clamp_min(1e-6)
    elif mode == "attribute": raw = ac
    elif mode == "object": raw = oc
    else: raise ValueError(mode)
    return (torch.log1p(raw) - torch.log1p(raw).min()) / (torch.log1p(raw).max() - torch.log1p(raw).min()).clamp_min(1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--logits", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--center", choices=("median", "mean"), required=True)
    ap.add_argument("--distance", choices=("abs", "rms"), required=True)
    ap.add_argument("--confidence", choices=("margin_high", "entropy_low"), required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--confidence-power", type=float, required=True)
    ap.add_argument("--delta", choices=("full", "update"), required=True)
    args = ap.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    paths = [Path(x).resolve() for x in args.logits]
    scores = torch.stack([zscore(torch.load(x, map_location="cpu", weights_only=True)["val"].float()) for x in paths])
    source = source_posterior(scores, args.center, args.distance, args.tau, args.confidence, args.confidence_power)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    delta = source if args.delta == "full" else source - text_logits
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    for support_name in ("min", "harmonic", "attribute", "object"):
        support = support_gate(cache, support_name)
        for exponent in (0.0, 0.25, 0.5, 1.0):
            s = support.clamp_min(1e-6).pow(exponent)
            for blend in ("product", "linear"):
                gate = s[None] if blend == "product" else 0.5 + 0.5 * s[None]
                gate = gate / gate.mean().clamp_min(1e-6)
                for strength in (0.5, 0.75, 1.0, 1.25, 1.5):
                    fused = evaluator(val["logits"].float() + strength * gate * delta, val["attr"], val["obj"], val["pair"])
                    rows.append({"support": support_name, "support_power": exponent, "blend": blend, "strength": strength, "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    best = max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"]))
    result = {"mode": "validation_only_graph_conditioned_adaptive_source_posterior", "test_read": False, "cache": {"path": str(cache_path), "sha256": sha256(cache_path)}, "sources": [{"path": str(x), "sha256": sha256(x)} for x in paths], "source_posterior": {"center": args.center, "distance": args.distance, "confidence": args.confidence, "tau": args.tau, "confidence_power": args.confidence_power, "delta": args.delta}, "base_val": base, "candidate_grid": {"support": ["min", "harmonic", "attribute", "object"], "support_power": [0.0, 0.25, 0.5, 1.0], "blend": ["product", "linear"], "strength": [0.5, 0.75, 1.0, 1.25, 1.5], "selection": "retain validation AUC gain >= 0.002, maximize validation HM"}, "best_balanced": best, "best_auc": best_auc, "rows": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": best, "best_auc": best_auc}, indent=2))


if __name__ == "__main__":
    main()
