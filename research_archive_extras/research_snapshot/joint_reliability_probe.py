"""Validation-only joint graph-support and CAMS-confidence gate probe."""

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


def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def support(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    ai = {x: i for i, x in enumerate(meta["attrs"])}
    oi = {x: i for i, x in enumerate(meta["objs"])}
    pa = torch.tensor([ai[a] for a, _ in meta["pairs"]])
    po = torch.tensor([oi[o] for _, o in meta["pairs"]])
    tp = cache["train"]["pair"].long()
    ac = torch.bincount(pa[tp], minlength=len(ai)).float()[pa]
    oc = torch.bincount(po[tp], minlength=len(oi)).float()[po]
    if mode == "min":
        raw = torch.minimum(ac, oc)
    elif mode == "harmonic":
        raw = 2 * ac * oc / (ac + oc).clamp_min(1e-6)
    elif mode == "object":
        raw = oc
    elif mode == "attribute":
        raw = ac
    else:
        raise ValueError(mode)
    return norm(torch.log1p(raw))


def conf_gate(logits: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(logits, 2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    p = torch.softmax(logits, dim=1)
    ent = -(p * p.clamp_min(1e-8).log()).sum(1)
    vals = {
        "entropy_low": 1 - norm(ent),
        "entropy_high": norm(ent),
        "margin_low": 1 - norm(margin),
        "margin_high": norm(margin),
    }
    return vals[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--logits", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    # Validation-only: do not access cache['test'] or logits['test'].
    val = cache["val"]
    paths = [Path(p).resolve() for p in args.logits]
    scores = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in paths])
    source = scores.median(0).values
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_logits = zscore(val["logits"].float())
    delta = source - text_logits
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    # Fixed grid declared before scoring; candidate selection is validation HM
    # after an AUC reliability gate.
    for support_name in ("min", "harmonic", "object", "attribute"):
        s = support(cache, support_name)
        for confidence_name in ("entropy_low", "entropy_high", "margin_low", "margin_high"):
            c = conf_gate(base_logits, confidence_name)
            for sp in (0.25, 0.5, 1.0):
                sg = s.clamp_min(1e-6).pow(sp)
                for cp in (0.25, 0.5, 1.0):
                    cg = c.clamp_min(1e-6).pow(cp)
                    for combine in ("product", "linear"):
                        gate = sg[None] * cg[:, None] if combine == "product" else 0.5 * sg[None] + 0.5 * cg[:, None]
                        gate = gate / gate.mean().clamp_min(1e-6)
                        for strength in (0.25, 0.5, 0.75, 1.0, 1.25):
                            fused = evaluator(val["logits"].float() + strength * gate * delta,
                                              val["attr"], val["obj"], val["pair"])
                            rows.append({"support": support_name, "confidence": confidence_name,
                                         "support_power": sp, "confidence_power": cp,
                                         "combine": combine, "strength": strength,
                                         "auc_gain": fused["AUC"] - base["AUC"],
                                         "hm_gain": fused["best_hm"] - base["best_hm"],
                                         "fused_val": fused})
    eligible = [r for r in rows if r["auc_gain"] >= 0.002]
    result = {
        "mode": "validation_only_joint_reliability",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in paths],
        "base_val": base,
        "candidate_grid": {"support": ["min", "harmonic", "object", "attribute"],
                            "confidence": ["entropy_low", "entropy_high", "margin_low", "margin_high"],
                            "support_power": [0.25, 0.5, 1.0], "confidence_power": [0.25, 0.5, 1.0],
                            "combine": ["product", "linear"], "strength": [0.25, 0.5, 0.75, 1.0, 1.25],
                            "selection": "retain validation AUC gain >= 0.002, maximize validation HM"},
        "best_balanced": max(eligible, key=lambda r: (r["hm_gain"], r["auc_gain"])) if eligible else None,
        "best_auc": max(rows, key=lambda r: (r["auc_gain"], r["hm_gain"])),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": result["best_balanced"], "best_auc": result["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
