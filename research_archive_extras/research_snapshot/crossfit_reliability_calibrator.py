"""Cross-fit, label-aware reliability calibration for a frozen residual expert.

The residual expert is fixed before this probe.  A tiny linear calibrator is
fit on four validation folds to predict when the expert's top-1 correction is
useful, then applied to the held-out fold.  No test tensor is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import CompositionDataset  # noqa: E402
from research.fast_feature_evaluation import FastClosedWorldEvaluator, zscore  # noqa: E402
from research.residual_mixture_probe import support_delta  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.amin(1, keepdim=True)) / (x.amax(1, keepdim=True) - x.amin(1, keepdim=True)).clamp_min(1e-6)


def image_features(base_logits: torch.Tensor, source_scores: torch.Tensor, pred_pair: torch.Tensor,
                   support_values: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(base_logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    probs = torch.softmax(base_logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
    median = source_scores.median(0).values
    dispersion = (source_scores - median).abs().mean((0, 2))
    source_margin = torch.topk(median, k=2, dim=1).values
    source_margin = source_margin[:, 0] - source_margin[:, 1]
    # The support lookup uses the base prediction, which is available at test time.
    support_pred = support_values.mean(0)[pred_pair]
    feats = torch.stack([
        row_norm(base_logits).amax(1),
        margin,
        entropy,
        source_margin,
        dispersion,
        support_pred,
    ], dim=1)
    return (feats - feats.mean(0, keepdim=True)) / feats.std(0, keepdim=True).clamp_min(1e-6)


def fit_logistic(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=0.03, weight_decay=0.1)
    positive = y.mean().clamp(0.05, 0.95)
    pos_weight = ((1.0 - positive) / positive).detach()
    for _ in range(300):
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return w.detach(), b.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--logits", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache_path = Path(args.cache).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val = cache["val"]
    source_paths = [Path(v).resolve() for v in args.logits]
    sources = torch.stack([zscore(torch.load(p, map_location="cpu", weights_only=True)["val"].float()) for p in source_paths])
    text = F.normalize(cache["prompts"]["pair"].float(), dim=-1)
    text_logits = zscore(val["com"].float() @ text.T)
    base_logits = val["logits"].float()
    source_delta, support_values = support_delta(
        sources, text_logits, cache,
        {"support_mode": "harmonic_image_support", "consensus": "inverse", "tau": 2.0, "support_exponent": 0.25, "blend": "linear"},
    )
    residual = 0.75 * source_delta
    candidate_logits = base_logits + residual
    base_pred = base_logits.argmax(1)
    candidate_pred = candidate_logits.argmax(1)
    truth = val["pair"].long()
    base_correct = base_pred.eq(truth)
    candidate_correct = candidate_pred.eq(truth)
    y = candidate_correct.float()
    x = image_features(base_logits, sources, base_pred, support_values)
    evaluator = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    base = evaluator(base_logits, val["attr"], val["obj"], val["pair"])
    raw = evaluator(candidate_logits, val["attr"], val["obj"], val["pair"])
    n = len(val["pair"])
    folds = torch.arange(n) % 5
    gate = torch.zeros(n)
    fold_stats = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        w, b = fit_logistic(x[train], y[train])
        gate[test] = torch.sigmoid(x[test] @ w + b)
        fold_stats.append({"fold": fold, "train_size": int(train.sum()), "test_size": int(test.sum()), "positive_rate": float(y[train].mean())})
    calibrated = evaluator(base_logits + gate[:, None] * residual, val["attr"], val["obj"], val["pair"])
    result = {
        "mode": "validation_only_5fold_crossfit_reliability_calibrator",
        "test_read": False,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in source_paths],
        "expert": {"support_mode": "harmonic_image_support", "consensus": "inverse", "tau": 2.0, "support_exponent": 0.25, "blend": "linear", "strength": 0.75},
        "calibrator": {"model": "linear_logistic", "features": ["base_peak", "base_margin", "base_entropy", "source_margin", "source_dispersion", "predicted_pair_support"], "folds": 5, "optimizer": "Adam lr=0.03 weight_decay=0.1, 300 steps", "selection": "fixed; no hyperparameter search"},
        "base_val": base,
        "raw_expert_val": raw,
        "crossfit_calibrated_val": calibrated,
        "gains": {"raw_auc": raw["AUC"] - base["AUC"], "raw_hm": raw["best_hm"] - base["best_hm"], "calibrated_auc": calibrated["AUC"] - base["AUC"], "calibrated_hm": calibrated["best_hm"] - base["best_hm"]},
        "folds": fold_stats,
        "gate_summary": {"mean": float(gate.mean()), "std": float(gate.std()), "min": float(gate.min()), "max": float(gate.max())},
        "label_summary": {"base_accuracy": float(base_correct.float().mean()), "expert_accuracy": float(candidate_correct.float().mean()), "expert_only_when_base_wrong": int((candidate_correct & ~base_correct).sum())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gains": result["gains"], "gate_summary": result["gate_summary"], "label_summary": result["label_summary"]}, indent=2))


if __name__ == "__main__":
    main()
