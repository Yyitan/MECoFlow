"""Protocol-locked fixed test for the trained interaction hyper-classifier."""

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
from research.interaction_hyper_classifier import InteractionHyperClassifier, classifier_logits  # noqa: E402
from research.warmstart_flow_classifier import pair_maps  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_fixed_test":
        raise ValueError("protocol not preregistered")
    cache_path = Path(protocol["cache"]).resolve()
    model_path = Path(protocol["model"]).resolve()
    validation_path = Path(protocol["validation_result"]).resolve()
    for path, key in ((cache_path, "cache_sha256"), (model_path, "model_sha256"), (validation_path, "validation_result_sha256")):
        if sha256(path) != protocol[key]:
            raise ValueError(f"SHA mismatch: {path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("test_read") is not False:
        raise ValueError("validation result is not test-free")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    pair_attr, pair_obj = pair_maps(cache)
    model = InteractionHyperClassifier(
        cache["prompts"]["pair"].float().to(device),
        cache["prompts"]["attr"].float().to(device),
        cache["prompts"]["obj"].float().to(device),
        pair_attr.to(device), pair_obj.to(device), int(protocol["rank"]),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    val = {k: v.to(device) for k, v in cache["val"].items()}
    test = {k: v.to(device) for k, v in cache["test"].items()}
    val_classifier = classifier_logits(model, cache["val"]["com"], device)
    test_classifier = classifier_logits(model, cache["test"]["com"], device)
    val_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="val"), device)
    test_eval = FastClosedWorldEvaluator(CompositionDataset(args.dataset_path, phase="test"), device)
    base_val = val_eval(val["logits"].float(), val["attr"], val["obj"], val["pair"])
    base_test = test_eval(test["logits"].float(), test["attr"], test["obj"], test["pair"])
    strength = float(protocol["candidate"]["strength"])
    delta_name = protocol["candidate"]["delta"]
    val_classifier = zscore(val_classifier.float())
    test_classifier = zscore(test_classifier.float())
    val_delta = val_classifier.to(device) if delta_name == "full" else val_classifier.to(device) - val["com"].float() @ torch.nn.functional.normalize(cache["prompts"]["pair"].float().to(device), dim=-1).T
    test_delta = test_classifier.to(device) if delta_name == "full" else test_classifier.to(device) - test["com"].float() @ torch.nn.functional.normalize(cache["prompts"]["pair"].float().to(device), dim=-1).T
    fused_val = val_eval(val["logits"].float() + strength * val_delta, val["attr"], val["obj"], val["pair"])
    fused_test = test_eval(test["logits"].float() + strength * test_delta, test["attr"], test["obj"], test["pair"])
    result = {
        "status": "passed", "test_read": True, "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "candidate": protocol["candidate"], "base_val": base_val, "fused_val": fused_val,
        "val_gain": {"AUC": fused_val["AUC"] - base_val["AUC"], "HM": fused_val["best_hm"] - base_val["best_hm"]},
        "base_test": base_test, "fused_test": fused_test,
        "test_gain": {"AUC": fused_test["AUC"] - base_test["AUC"], "HM": fused_test["best_hm"] - base_test["best_hm"]},
        "model": {"path": str(model_path), "sha256": sha256(model_path)}, "test_search_after_lock": False,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"val_gain": result["val_gain"], "test_gain": result["test_gain"]}, indent=2))


if __name__ == "__main__":
    main()
