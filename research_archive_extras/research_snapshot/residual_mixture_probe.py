"""Validation-only mixture of locked source-posterior and support residuals.

The two component residuals are fixed before this probe is run.  Only the
convex mixing coefficient is screened on validation; cache/source test tensors
are never accessed.
"""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_norm(value: torch.Tensor) -> torch.Tensor:
    lo = value.amin(dim=1, keepdim=True)
    hi = value.amax(dim=1, keepdim=True)
    return (value - lo) / (hi - lo).clamp_min(1e-6)


def support_gate(cache: dict[str, object], mode: str) -> torch.Tensor:
    meta = cache["meta"]
    attrs = {name: i for i, name in enumerate(meta["attrs"])}
    objs = {name: i for i, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    attr = attr_count[pair_attr]
    obj = obj_count[pair_obj]
    if mode == "min_image_support":
        raw = torch.minimum(attr, obj)
    elif mode == "harmonic_image_support":
        raw = 2.0 * attr * obj / (attr + obj).clamp_min(1e-6)
    else:
        raise ValueError(mode)
    raw = torch.log1p(raw)
    return (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-6)


def adaptive_delta(scores: torch.Tensor, text_logits: torch.Tensor, cfg: dict[str, object]):
    center = scores.median(0).values if cfg["center"] == "median" else scores.mean(0)
    residual = scores - center[None]
    if cfg["distance"] == "abs":
        distance = residual.abs().mean(2)
    elif cfg["distance"] == "rms":
        distance = residual.square().mean(2).sqrt()
    else:
        raise ValueError(cfg["distance"])
    distance = distance / distance.mean(1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-float(cfg["tau"]) * distance)
    top2 = torch.topk(scores, k=2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    probs = torch.softmax(scores, dim=2)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(2)
    if cfg["confidence"] == "margin_high":
        conf = row_norm(margin)
    elif cfg["confidence"] == "entropy_low":
        conf = 1.0 - row_norm(entropy)
    elif cfg["confidence"] == "none":
        conf = torch.ones(scores.shape[:2])
    else:
        raise ValueError(cfg["confidence"])
    weights = weights * conf.clamp_min(1e-6).pow(float(cfg["confidence_power"]))
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    posterior = (weights[:, :, None] * scores).sum(0)
    delta = posterior if cfg["delta"] == "full" else posterior - text_logits
    return delta, weights


def support_delta(scores: torch.Tensor, text_logits: torch.Tensor,
                  cache: dict[str, object], cfg: dict[str, object]):
    median = scores.median(0).values
    dispersion = (scores - median).abs().mean(0)
    scale = dispersion.median(1, keepdim=True).values.clamp_min(1e-6)
    relative = dispersion / scale
    if cfg["consensus"] == "none":
        agreement = torch.ones_like(relative)
    elif cfg["consensus"] == "inverse":
        agreement = 1.0 / (1.0 + float(cfg["tau"]) * relative)
    else:
        raise ValueError(cfg["consensus"])
    agreement = agreement / agreement.mean(1, keepdim=True).clamp_min(1e-6)
    support = support_gate(cache, cfg["support_mode"])
    support_power = support.clamp_min(1e-6).pow(float(cfg["support_exponent"]))
    if cfg["blend"] == "linear":
        gate = 0.5 * agreement + 0.5 * support_power[None]
    elif cfg["blend"] == "geometric":
        gate = agreement * support_power[None]
    else:
        raise ValueError(cfg["blend"])
    gate = gate / gate.mean(1, keepdim=True).clamp_min(1e-6)
    return gate * (median - text_logits), gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adaptive", required=True, help="JSON object for locked adaptive candidate")
    parser.add_argument("--support", required=True, help="JSON object for locked support candidate")
    parser.add_argument("--adaptive-strength", type=float, required=True)
    parser.add_argument("--support-strength", type=float, required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    source_paths = [Path(value).resolve() for value in args.logits]
    scores = torch.stack([
        zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float())
        for path in source_paths
    ])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    adaptive_arg = Path(args.adaptive)
    support_arg = Path(args.support)
    adaptive_cfg = json.loads(adaptive_arg.read_text(encoding="utf-8")) if adaptive_arg.exists() else json.loads(args.adaptive)
    support_cfg = json.loads(support_arg.read_text(encoding="utf-8")) if support_arg.exists() else json.loads(args.support)
    adaptive, adaptive_weights = adaptive_delta(scores, text_logits, adaptive_cfg)
    support, support_gate_values = support_delta(scores, text_logits, cache, support_cfg)
    evaluator = FastClosedWorldEvaluator(
        CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu")
    )
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    rows = []
    # The only searched quantity is the convex mixture coefficient.
    alphas = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    for alpha in alphas:
        delta = (alpha * float(args.adaptive_strength) * adaptive
                 + (1.0 - alpha) * float(args.support_strength) * support)
        fused = evaluator(
            val["logits"].float() + delta,
            val["attr"], val["obj"], val["pair"],
        )
        rows.append({
            "alpha_adaptive": alpha,
            "alpha_support": 1.0 - alpha,
            "auc_gain": fused["AUC"] - base["AUC"],
            "hm_gain": fused["best_hm"] - base["best_hm"],
            "fused_val": fused,
        })
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    best_balanced = max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None
    best_auc = max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"]))
    result = {
        "mode": "validation_only_locked_reliability_residual_mixture",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "base_val": base,
        "components": {
            "adaptive": {"config": adaptive_cfg, "strength": float(args.adaptive_strength)},
            "support": {"config": support_cfg, "strength": float(args.support_strength)},
        },
        "candidate_grid": {
            "alpha_adaptive": list(alphas),
            "selection": "retain validation AUC gain >= 0.002, maximize validation HM",
        },
        "best_balanced": best_balanced,
        "best_auc": best_auc,
        "rows": rows,
        "diagnostics": {
            "adaptive_weight_mean": adaptive_weights.mean(1).tolist(),
            "support_gate_mean": support_gate_values.mean(1).tolist(),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "base_val": base,
        "best_balanced": None if best_balanced is None else {
            k: best_balanced[k] for k in ("auc_gain", "hm_gain", "alpha_adaptive", "alpha_support")
        },
        "best_auc": {k: best_auc[k] for k in ("auc_gain", "hm_gain", "alpha_adaptive", "alpha_support")},
    }, indent=2))


if __name__ == "__main__":
    main()
