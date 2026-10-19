"""Protocol-locked fixed test for an image-level confidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import sys
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


def normalize(value: torch.Tensor) -> torch.Tensor:
    return (value - value.min()) / (value.max() - value.min()).clamp_min(1e-6)


def confidence(logits: torch.Tensor, name: str) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(1)
    values = {
        "base_entropy_low": 1.0 - normalize(entropy),
        "base_entropy_high": normalize(entropy),
        "base_margin_high": normalize(margin),
        "text_entropy_low": 1.0 - normalize(entropy),
        "text_entropy_high": normalize(entropy),
        "text_margin_high": normalize(margin),
        "text_margin_low": 1.0 - normalize(margin),
    }
    return values[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = (root / protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol must be preregistered")
    candidates = protocol.get("candidate_set", [])
    if len(candidates) != 1:
        raise ValueError("exactly one candidate required")
    candidate = candidates[0]
    allowed = {
        ("base_entropy_low", 0.5, 1.0),
        ("base_entropy_high", 0.25, 0.5),
    }
    signature = (candidate["confidence"], float(candidate["power"]), float(candidate["strength"]))
    if signature not in allowed:
        raise ValueError(f"unexpected candidate {signature!r}")
    if "do not search" not in protocol.get("test_rule", ""):
        raise ValueError("test rule does not prohibit search")
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()
    cache_path = resolve(protocol["cache"])
    if sha256(cache_path) != protocol["cache_sha256"]:
        raise ValueError("cache sha mismatch")
    source_paths = [resolve(item["path"]) for item in protocol["sources"]]
    for path, item in zip(source_paths, protocol["sources"]):
        if sha256(path) != item["sha256"]:
            raise ValueError(f"source sha mismatch: {path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    val, test = cache["val"], cache["test"]
    saved = [torch.load(path, map_location="cpu", weights_only=True) for path in source_paths]
    text = torch.nn.functional.normalize(cache["prompts"]["pair"].float(), dim=-1)
    val_text = zscore(val["com"].float() @ text.T)
    test_text = zscore(test["com"].float() @ text.T)
    val_base_z, test_base_z = zscore(val["logits"].float()), zscore(test["logits"].float())
    val_source = torch.stack([zscore(item["val"].float()) for item in saved]).median(0).values
    test_source = torch.stack([zscore(item["test"].float()) for item in saved]).median(0).values
    # The candidate names are intentionally explicit; base confidence uses CAMS
    # logits, while text names use CLIP pair logits.
    val_conf_logits = val_base_z if candidate["confidence"].startswith("base_") else val_text
    test_conf_logits = test_base_z if candidate["confidence"].startswith("base_") else test_text
    val_gate = confidence(val_conf_logits, candidate["confidence"]).clamp_min(1e-6).pow(float(candidate["power"]))
    test_gate = confidence(test_conf_logits, candidate["confidence"]).clamp_min(1e-6).pow(float(candidate["power"]))
    val_gate = val_gate / val_gate.mean().clamp_min(1e-6)
    test_gate = test_gate / test_gate.mean().clamp_min(1e-6)
    val_delta, test_delta = val_source - val_text, test_source - test_text
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), torch.device("cpu"))
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), torch.device("cpu"))
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = float(candidate["strength"])
    fused_val = val_eval(val["logits"].float() + strength * val_gate[:, None] * val_delta, val["attr"], val["obj"], val["pair"])
    fused_test = test_eval(test["logits"].float() + strength * test_gate[:, None] * test_delta, test["attr"], test["obj"], test["pair"])
    individual = []
    for index, (val_item, test_item) in enumerate(zip(saved, saved)):
        iv = val_eval(val["logits"].float() + strength * val_gate[:, None] * (zscore(val_item["val"].float()) - val_text), val["attr"], val["obj"], val["pair"])
        it = test_eval(test["logits"].float() + strength * test_gate[:, None] * (zscore(test_item["test"].float()) - test_text), test["attr"], test["obj"], test["pair"])
        individual.append({"seed_index": index, "test_auc_gain": it["AUC"] - base_test["AUC"], "test_hm_gain": it["best_hm"] - base_test["best_hm"], "val_auc_gain": iv["AUC"] - base_val["AUC"], "val_hm_gain": iv["best_hm"] - base_val["best_hm"]})
    result = {
        "status": "passed",
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path), "candidate": candidate, "cache": str(cache_path), "cache_sha256": sha256(cache_path), "sources": [{"path": str(path), "sha256": sha256(path)} for path in source_paths], "selection_locked": True, "test_read": True},
        "base_val": base_val, "fused_val": fused_val, "base_test": base_test, "fused_test": fused_test, "individual": individual,
        "mean": {"val_auc_gain": fused_val["AUC"] - base_val["AUC"], "val_hm_gain": fused_val["best_hm"] - base_val["best_hm"], "test_auc_gain": fused_test["AUC"] - base_test["AUC"], "test_hm_gain": fused_test["best_hm"] - base_test["best_hm"], "positive_individual_test_auc_runs": sum(row["test_auc_gain"] > 0 for row in individual), "positive_individual_test_hm_runs": sum(row["test_hm_gain"] > 0 for row in individual)},
        "test_search_after_lock": False,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
