"""Protocol-locked fixed test for adaptive non-parametric source weighting."""

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_norm(value: torch.Tensor) -> torch.Tensor:
    lo = value.amin(dim=1, keepdim=True)
    hi = value.amax(dim=1, keepdim=True)
    return (value - lo) / (hi - lo).clamp_min(1e-6)


def confidence(scores: torch.Tensor, mode: str) -> torch.Tensor:
    top2 = torch.topk(scores, k=2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]
    probabilities = torch.softmax(scores, dim=2)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(2)
    if mode == "none":
        return torch.ones(scores.shape[:2])
    if mode == "margin_high":
        return row_norm(margin)
    if mode == "entropy_low":
        return 1.0 - row_norm(entropy)
    raise ValueError(mode)


def posterior(scores: torch.Tensor, candidate: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    center = scores.median(0).values if candidate["center"] == "median" else scores.mean(0)
    residual = scores - center[None]
    if candidate["distance"] == "abs":
        distance = residual.abs().mean(2)
    elif candidate["distance"] == "rms":
        distance = residual.square().mean(2).sqrt()
    else:
        raise ValueError(candidate["distance"])
    distance = distance / distance.mean(1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-float(candidate["tau"]) * distance)
    weights = weights * confidence(scores, candidate["confidence"]).clamp_min(1e-6).pow(float(candidate["confidence_power"]))
    weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
    return (weights[:, :, None] * scores).sum(0), weights


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path = resolve(args.protocol, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered")
    candidates = protocol.get("candidate_set", [])
    if len(candidates) != 1:
        raise ValueError("exactly one candidate required")
    candidate = candidates[0]
    required = {"center", "distance", "confidence", "tau", "confidence_power", "delta", "strength"}
    if set(candidate) != required:
        raise ValueError(f"candidate keys mismatch: {sorted(candidate)}")
    if "validation AUC gain" not in protocol.get("validation_selection_rule", "") or "largest validation HM" not in protocol.get("validation_selection_rule", ""):
        raise ValueError("validation selection rule is incomplete")
    if "do not search" not in protocol.get("test_rule", ""):
        raise ValueError("test rule does not prohibit search")
    if protocol.get("validation_gate", {}).get("test_read") is not False:
        raise ValueError("validation gate must be test-free")

    cache_path = resolve(protocol["cache"], root)
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache sha mismatch")
    source_paths = [resolve(item["path"], root) for item in protocol["sources"]]
    for path, item in zip(source_paths, protocol["sources"]):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source sha mismatch: {path}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    val_scores = torch.stack([zscore(item["val"].float()) for item in saved])
    test_scores = torch.stack([zscore(item["test"].float()) for item in saved])
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_source, val_weights = posterior(val_scores, candidate)
    test_source, test_weights = posterior(test_scores, candidate)
    val_delta = val_source if candidate["delta"] == "full" else val_source - val_text
    test_delta = test_source if candidate["delta"] == "full" else test_source - test_text
    evaluator_val = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    evaluator_test = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = evaluator_val(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = evaluator_test(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = float(candidate["strength"])
    fused_val = evaluator_val(val["logits"].float() + strength * val_delta, val["attr"], val["obj"], val["pair"])
    fused_test = evaluator_test(test["logits"].float() + strength * test_delta, test["attr"], test["obj"], test["pair"])
    val_auc_gain = fused_val["AUC"] - base_val["AUC"]
    val_hm_gain = fused_val["best_hm"] - base_val["best_hm"]
    expected = protocol["validation_gate"]
    if abs(val_auc_gain - float(expected["mean_auc_gain"])) > 1e-10 or abs(val_hm_gain - float(expected["mean_hm_gain"])) > 1e-10:
        raise ValueError(f"validation replay mismatch: {(val_auc_gain, val_hm_gain)} vs {(expected['mean_auc_gain'], expected['mean_hm_gain'])}")
    result = {
        "status": "passed",
        "protocol": {
            "path": str(protocol_path), "sha256": sha256(protocol_path), "candidate": candidate,
            "cache": str(cache_path), "cache_sha256": sha256(cache_path),
            "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
            "selection_locked": True, "test_read": True,
        },
        "base_val": base_val, "fused_val": fused_val,
        "base_test": base_test, "fused_test": fused_test,
        "weights": {"val_mean": val_weights.mean(1).tolist(), "test_mean": test_weights.mean(1).tolist()},
        "source_count": len(source_paths),
        "mean": {
            "val_auc_gain": val_auc_gain, "val_hm_gain": val_hm_gain,
            "test_auc_gain": fused_test["AUC"] - base_test["AUC"],
            "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"],
        },
        "test_search_after_lock": False,
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
