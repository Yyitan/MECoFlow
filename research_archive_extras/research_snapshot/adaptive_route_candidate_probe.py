"""Validation-only instance-adaptive blend of frozen transport routes.

The two route residuals are frozen before this probe.  Only a small, explicit
family of per-image blend schedules is searched on validation, and the output
contains no test tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def support_weights(cache: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = cache["meta"]
    attrs = {name: index for index, name in enumerate(meta["attrs"])}
    objs = {name: index for index, name in enumerate(meta["objs"])}
    pair_attr = torch.tensor([attrs[a] for a, _ in meta["pairs"]])
    pair_obj = torch.tensor([objs[o] for _, o in meta["pairs"]])
    train_pair = cache["train"]["pair"].long()
    attr_count = torch.bincount(pair_attr[train_pair], minlength=len(attrs)).float()
    obj_count = torch.bincount(pair_obj[train_pair], minlength=len(objs)).float()
    pair = normalize(torch.log1p(torch.minimum(attr_count[pair_attr], obj_count[pair_obj])))
    pair_seen = torch.bincount(train_pair, minlength=len(meta["pairs"])) > 0
    return pair, pair_seen


def entropy_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return normalize(entropy)


def margin_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.float().topk(2, dim=1).values
    return 1.0 - normalize(top2[:, 0] - top2[:, 1])


def schedules(
    entropy: torch.Tensor,
    margin: torch.Tensor,
    support: torch.Tensor,
    seen: torch.Tensor,
    pair_score: torch.Tensor,
    pair_seen: torch.Tensor,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Constant schedules include the already-frozen .5 blend as a control.
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows.append({"name": f"constant_{alpha:.2f}", "alpha": torch.full_like(entropy, alpha)})
    for signal_name, signal in (("entropy", entropy), ("margin", margin)):
        for power in (0.5, 1.0, 2.0):
            rows.append({"name": f"{signal_name}_power_{power:g}", "alpha": signal.clamp(0, 1).pow(power)})
    for signal_name, signal in (("entropy", entropy), ("margin", margin)):
        for power in (0.5, 1.0, 2.0):
            # Support controls how much the HM-specialist route is trusted;
            # uncertainty resolves image-level ambiguity inside that prior.
            alpha = 0.5 * support.clamp_min(1e-6).pow(power) + 0.5 * signal
            rows.append({"name": f"support_plus_{signal_name}_{power:g}", "alpha": alpha.clamp(0, 1)})
            alpha = 0.5 * (1.0 - support).clamp_min(1e-6).pow(power) + 0.5 * signal
            rows.append({"name": f"inverse_support_plus_{signal_name}_{power:g}", "alpha": alpha.clamp(0, 1)})
    for power in (0.5, 1.0, 2.0):
        alpha = seen + (1.0 - seen) * entropy.clamp(0, 1).pow(power)
        rows.append({"name": f"unseen_entropy_{power:g}", "alpha": alpha.clamp(0, 1)})
    # Candidate-conditioned gate: route each candidate pair using its own frozen
    # CAMS confidence, while keeping seen-pair scores on the HM specialist.
    for slope in (-8.0, -4.0, 4.0, 8.0):
        for threshold in (-0.5, 0.0, 0.5):
            alpha = torch.sigmoid(slope * (pair_score - threshold))
            alpha = torch.where(pair_seen[None], torch.ones_like(alpha), alpha)
            rows.append({
                "name": f"candidate_score_sigmoid_{slope:g}_{threshold:g}",
                "alpha": alpha,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--iso-logits", nargs=3, required=True)
    parser.add_argument("--diag-logits", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dataset import CompositionDataset
    from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore

    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    iso_paths = [Path(path).resolve() for path in args.iso_logits]
    diag_paths = [Path(path).resolve() for path in args.diag_logits]
    iso = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in iso_paths]).mean(0)
    diag = torch.stack([zscore(torch.load(path, map_location="cpu", weights_only=True)["val"].float()) for path in diag_paths]).mean(0)
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    support, pair_seen = support_weights(cache)
    # The schedule is defined on graph-seen/unseen composition pairs.  Use the
    # boolean pair mask, not real_centroids' image-count vector; counts would
    # produce invalid alpha values greater than one on seen pairs.
    seen = pair_seen[val["pair"].long()].float()
    uncertainty = entropy_uncertainty(val["logits"])
    margin = margin_uncertainty(val["logits"])
    pair_score = (val["logits"].float() - val["logits"].float().mean(1, keepdim=True)) / val["logits"].float().std(1, keepdim=True).clamp_min(1e-6)
    # A is the frozen AUC specialist; B is the frozen balanced specialist.
    inverse = (1.0 - support).clamp_min(1e-6)[None] * diag + support[None] * iso
    soft = support.clamp_min(1e-6).pow(0.25)[None] * diag + (1.0 - support.clamp_min(1e-6).pow(0.25))[None] * iso
    soft_scope = pair_seen.float() + (1.0 - pair_seen.float()) * 0.25
    delta_a = inverse - text_logits
    delta_b = soft * soft_scope[val["pair"].long()][:, None] * 1.5
    rows = []
    for item in schedules(uncertainty, margin, support[val["pair"].long()], seen, pair_score, pair_seen):
        alpha = item["alpha"]
        alpha_broadcast = alpha[:, None] if alpha.ndim == 1 else alpha
        delta = (1.0 - alpha_broadcast) * delta_a + alpha_broadcast * delta_b
        fused = evaluator(val["logits"].float() + delta, val["attr"], val["obj"], val["pair"])
        rows.append({"name": item["name"], "alpha_shape": list(alpha.shape), "mean_alpha": float(alpha.mean()), "auc_gain": fused["AUC"] - base["AUC"], "hm_gain": fused["best_hm"] - base["best_hm"], "fused_val": fused})
    eligible = [row for row in rows if row["auc_gain"] >= 0.002]
    result = {
        "mode": "validation_only_instance_adaptive_route_blend",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(path), "sha256": sha256(path)} for path in iso_paths + diag_paths],
        "frozen_candidates": {"A": "inverse_pair_support^1/update x 1.0", "B": "pair_support^0.25/seen_lambda=.25/full x 1.5"},
        "selection": "retain AUC gain >= .002 and maximize HM",
        "base_val": base,
        "best_balanced": max(eligible, key=lambda row: (row["hm_gain"], row["auc_gain"])) if eligible else None,
        "best_auc": max(rows, key=lambda row: (row["auc_gain"], row["hm_gain"])),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_balanced": result["best_balanced"], "best_auc": result["best_auc"]}, indent=2))


if __name__ == "__main__":
    main()
